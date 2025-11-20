import os
import shutil
import subprocess
from pathlib import Path
import torch
import tensorrt as trt
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from model import build_transformer



# Utility functions
def load_tokenizers(config):
    """Load source and target tokenizers."""
    if Tokenizer is None:
        raise ImportError("tokenizers library not available.")

    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))

    return (
        Tokenizer.from_file(str(tokenizer_src_path)),
        Tokenizer.from_file(str(tokenizer_tgt_path)),
    )


def prepare_model_variables(config):
    """Prepare model-related constants such as vocab sizes and sequence lengths."""
    tokenizer_src, tokenizer_tgt = load_tokenizers(config)

    return {
        "SRC_VOCAB_SIZE": tokenizer_src.get_vocab_size(),
        "TGT_VOCAB_SIZE": tokenizer_tgt.get_vocab_size(),
        "SRC_SEQ_LEN": config["seq_len"],
        "TGT_SEQ_LEN": config["seq_len"],
        "D_MODEL": config["d_model"],
        "N_LAYERS": 6,
        "N_HEADS": 8,
        "D_FF": 2048,
        "DROPOUT": 0.1,
        "tokenizer_src": tokenizer_src,
        "tokenizer_tgt": tokenizer_tgt,
    }


def export_onnx(model, onnx_path, shapes):
    """Export a PyTorch model to ONNX format."""
    if torch is None:
        raise ImportError("Torch is required to export ONNX.")

    dummy_src = torch.zeros(1, shapes["seq_len"], dtype=torch.long)
    dummy_tgt = torch.zeros(1, shapes["seq_len"], dtype=torch.long)
    dummy_src_mask = torch.ones(1, 1, shapes["seq_len"], shapes["seq_len"])
    dummy_tgt_mask = torch.ones(1, 1, shapes["seq_len"], shapes["seq_len"])

    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt, dummy_src_mask, dummy_tgt_mask),
        onnx_path,
        opset_version=17,
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


def build_trt_engine(onnx_path, engine_path, max_seq_len, use_fp16=True):
    """Build a TensorRT engine from ONNX using trtexec."""
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not shutil.which(trtexec):
        raise RuntimeError(f"trtexec not found at {trtexec}")

    min_shapes = f"--minShapes=src:1x1,tgt:1x1,src_mask:1x1x1x1,tgt_mask:1x1x1x1"
    opt_shapes = f"--optShapes=src:1x128,tgt:1x128,src_mask:1x1x128x128,tgt_mask:1x1x128x128"
    max_shapes = (
        f"--maxShapes=src:1x{max_seq_len},tgt:1x{max_seq_len},"
        f"src_mask:1x1x{max_seq_len}x{max_seq_len},tgt_mask:1x1x{max_seq_len}x{max_seq_len}"
    )

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        min_shapes,
        opt_shapes,
        max_shapes,
        "--verbose",
    ]

    if use_fp16:
        cmd.append("--fp16")

    subprocess.run(" ".join(cmd), shell=True, check=True)


def main():
    """Main workflow for exporting ONNX and TensorRT models."""
    config = get_config()

    # Prepare output dirs
    onnx_dir = Path("onnx")
    trt_dir = Path("tensorrt")
    onnx_dir.mkdir(exist_ok=True)
    trt_dir.mkdir(exist_ok=True)

    # Get latest checkpoint
    ckpt_path = latest_weights_file_path(config)
    if ckpt_path is None:
        raise RuntimeError("No checkpoint file found.")

    model_name = Path(ckpt_path).stem
    onnx_path = onnx_dir / f"{model_name}.onnx"
    engine_path = trt_dir / f"{model_name}.engine"

    vars = prepare_model_variables(config)

    # Build model
    model = build_transformer(
        vars["SRC_VOCAB_SIZE"],
        vars["TGT_VOCAB_SIZE"],
        vars["SRC_SEQ_LEN"],
        vars["TGT_SEQ_LEN"],
        d_model=vars["D_MODEL"],
        N=vars["N_LAYERS"],
        h=vars["N_HEADS"],
        dropout=vars["DROPOUT"],
        d_ff=vars["D_FF"],
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    # Export ONNX
    export_onnx(model, str(onnx_path), {"seq_len": vars["SRC_SEQ_LEN"]})

    # Build TensorRT engine
    build_trt_engine(str(onnx_path), str(engine_path), vars["SRC_SEQ_LEN"])

    print("\n🎉 Export complete!")
    print(f"ONNX file saved to: {onnx_path}")
    print(f"TensorRT engine saved to: {engine_path}")


if __name__ == "__main__":
    main()
