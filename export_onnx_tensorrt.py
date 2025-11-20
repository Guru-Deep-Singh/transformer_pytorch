import torch
import subprocess
import shutil
import os
import tensorrt as trt
from model import build_transformer
from config import get_config, latest_weights_file_path
from tokenizers import Tokenizer
from pathlib import Path

# Getting config and setting up tokenizers
config = get_config()

tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))

tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

# Setting variables for ONNX and TensorRT export
SRC_VOCAB_SIZE = tokenizer_src.get_vocab_size()
TGT_VOCAB_SIZE = tokenizer_tgt.get_vocab_size()
SRC_SEQ_LEN = config['seq_len'] # 512
TGT_SEQ_LEN = config['seq_len'] # 512
D_MODEL = config['d_model']
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

CHECKPOINT_PATH = latest_weights_file_path(config)
MODEL_STR = latest_weights_file_path(config).split("/")[-1].split(".")[0]


# Make ONNX directory
ONNX_DIR = "onnx"
os.makedirs(ONNX_DIR, exist_ok=True)
ONNX_PATH = f"{ONNX_DIR}/{MODEL_STR}.onnx"
print(f"ONNX will be saved to {ONNX_PATH}")


# TensorRT Engine Builder
TRT_DIR = "tensorrt"
os.makedirs(TRT_DIR, exist_ok=True)

# Jetson trtexec location
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
if not shutil.which(TRTEXEC):
    raise RuntimeError(f"trtexec not found at {TRTEXEC}")

# Function to build TensorRT engine
def build_trt_engine(onnx_path, model_name, max_seq_len, use_fp16=True):
    """Builds a TensorRT engine from ONNX using trtexec with dynamic shape support."""

    # Determine engine file name
    precision = "fp16" if use_fp16 else "fp32"
    engine_path = f"{TRT_DIR}/{model_name}_dynamic_{precision}.engine"

    print(f"\n➡ Building TensorRT engine: {engine_path}")
    print(f"   Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"   Dynamic Max Sequence Length: {max_seq_len}")

    # Define the range for dynamic shapes based on the ONNX dynamic axes:
    # src: 1xseq_len, tgt: 1xseq_len
    # src_mask: 1x1xseq_lenxseq_len, tgt_mask: 1x1xseq_lenxseq_len
    
    # We use a batch size of 1 and define the sequence length ranges (1 to max_seq_len)
    shape_ranges = (
        # Minimum shape (1 token)
        f"--minShapes=src:1x1,tgt:1x1,"
        f"src_mask:1x1x1x1,tgt_mask:1x1x1x1 "
        # Optimal shape (a typical working length, e.g., 128)
        f"--optShapes=src:1x128,tgt:1x128,"
        f"src_mask:1x1x128x128,tgt_mask:1x1x128x128 "
        # Maximum shape (512, defined in config)
        f"--maxShapes=src:1x{max_seq_len},tgt:1x{max_seq_len},"
        f"src_mask:1x1x{max_seq_len}x{max_seq_len},tgt_mask:1x1x{max_seq_len}x{max_seq_len}"
    )

    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        shape_ranges,
        "--verbose"
    ]

    # enable fp16
    if use_fp16:
        cmd.append("--fp16")

    print("Running command:")
    print(" ".join(cmd))

    
    model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, d_model=D_MODEL, N=N_LAYERS, h=N_HEADS, dropout=DROPOUT, d_ff=D_FF)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    
    batch_size = 1
    dummy_src = torch.zeros(batch_size, SRC_SEQ_LEN, dtype=torch.long)
    dummy_tgt = torch.zeros(batch_size, TGT_SEQ_LEN, dtype=torch.long)
    dummy_src_mask = torch.ones(batch_size, 1, SRC_SEQ_LEN, SRC_SEQ_LEN)
    dummy_tgt_mask = torch.ones(batch_size, 1, TGT_SEQ_LEN, TGT_SEQ_LEN)

    torch.onnx.export(model, (dummy_src, dummy_tgt, dummy_src_mask, dummy_tgt_mask), ONNX_PATH, opset_version=17,
        input_names=["src", "tgt", "src_mask", "tgt_mask"],
        output_names=["logits"],
        dynamic_axes={
            "src": {1: "src_len"},
            "tgt": {1: "tgt_len"},
            "src_mask": {2: "src_len", 3: "src_len"},
            "tgt_mask": {2: "tgt_len", 3: "tgt_len"},
            "logits": {1: "tgt_len"},
        },
    )
    print(f"Exported ONNX model to {ONNX_PATH}")

    subprocess.run(" ".join(cmd), shell=True, check=True)
    print(f"✅ TensorRT engine saved to: {engine_path}")
    return engine_path


# Build TensorRT engine
USE_FP16 = True
engine_path = build_trt_engine(ONNX_PATH, MODEL_STR, SRC_SEQ_LEN, use_fp16=USE_FP16)

print("\n🎉 Done! ONNX + TensorRT engine created successfully.")
print(f"ONNX file:   {ONNX_PATH}")
print(f"TensorRT engine: {engine_path}")