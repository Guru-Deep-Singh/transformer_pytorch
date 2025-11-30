import os
import shutil
import subprocess
from pathlib import Path
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from model import build_transformer
import gc


# WRAPPER CLASSES
# These wrappers allow us to export specific methods (encode, decode, project)
# as if they were the main 'forward' method, which ONNX requires.


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, src, src_mask):
        return self.model.encode(src, src_mask)

class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, encoder_output, src_mask, tgt, tgt_mask):
        return self.model.decode(encoder_output, src_mask, tgt, tgt_mask)

class ProjectionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model.project(x)


# UTILITIES

def load_tokenizers(config):
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    return Tokenizer.from_file(str(tokenizer_src_path)), Tokenizer.from_file(str(tokenizer_tgt_path))

def prepare_model_variables(config):
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
        "DROPOUT": 0.1
    }

def export_single_onnx(model_wrapper, dummy_inputs, onnx_path, input_names, output_names, dynamic_axes):
    """Generic function to export any wrapper to ONNX."""
    if torch is None: raise ImportError("Torch is required.")
    
    print(f"--> Exporting {onnx_path}...")
    torch.onnx.export(
        model_wrapper,
        dummy_inputs,
        onnx_path,
        opset_version=17, # High opset for better Transformer support
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False
    )

def build_single_engine(onnx_path, engine_path, min_shapes, opt_shapes, max_shapes, use_fp16=True):
    """Generic function to build TRT engine with specific shapes."""
    trtexec = "/usr/src/tensorrt/bin/trtexec" # Adjust path if needed
    if not shutil.which("trtexec") and not os.path.exists(trtexec):
        # Fallback check or error
        raise RuntimeError("trtexec not found. Please check PATH or variable.")

    if use_fp16:
        engine_path = engine_path.split(".")[0] + "_fp16.engine"
    else:
        engine_path = engine_path.split(".")[0] + "_fp32.engine"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        "--verbose"
    ]
    if use_fp16:
        cmd.append("--fp16")

    print(f"--> Building Engine {engine_path}...")
    subprocess.run(" ".join(cmd), shell=True, check=True)


# MAIN WORKFLOW

def main():
    config = get_config()
    onnx_dir = Path("onnx_split")
    trt_dir = Path("tensorrt_split")
    onnx_dir.mkdir(exist_ok=True)
    trt_dir.mkdir(exist_ok=True)

    ckpt_path = latest_weights_file_path(config)
    if ckpt_path is None: raise RuntimeError("No checkpoint found.")
    
    model_base_name = Path(ckpt_path).stem
    vars = prepare_model_variables(config)
    
    # Load Monolithic Model
    print("Loading PyTorch Model...")
    model = build_transformer(
        vars["SRC_VOCAB_SIZE"], vars["TGT_VOCAB_SIZE"],
        vars["SRC_SEQ_LEN"], vars["TGT_SEQ_LEN"],
        d_model=vars["D_MODEL"], N=vars["N_LAYERS"], h=vars["N_HEADS"],
        dropout=vars["DROPOUT"], d_ff=vars["D_FF"]
    )


    # When I trained a model I used custom LayerNormalization which had a problem with fp16
    # I replaced it with nn.LayerNorm thus need to have a patch to map weights with the trained values
    PATCH = False

    # Load the pretrained weights
    checkpoint  = torch.load(ckpt_path,  map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    if PATCH:
        # Rename 'alpha' to 'weight'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".alpha"):
                # Replace .alpha with .weight
                new_key = key.replace(".alpha", ".weight")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        del checkpoint, state_dict, new_state_dict
    else:
        model.load_state_dict(state_dict)
        del checkpoint, state_dict

    torch.cuda.empty_cache()
    gc.collect()

    model.eval()

    # Common Dimensions
    bs = 1
    seq_len = vars["SRC_SEQ_LEN"] # e.g., 350
    d_model = vars["D_MODEL"]     # e.g., 512
    USE_FP16 = True

    # ==========================================================================
    # ENCODER
    # ==========================================================================
    enc_wrapper = EncoderWrapper(model)
    enc_onnx = onnx_dir / f"{model_base_name}_encoder.onnx"
    enc_engine = trt_dir / f"{model_base_name}_encoder.engine"

    dummy_src = torch.randint(0, vars["SRC_VOCAB_SIZE"], (bs, seq_len))
    dummy_mask = torch.ones(bs, 1, seq_len, seq_len)

    export_single_onnx(
        enc_wrapper,
        (dummy_src, dummy_mask),
        str(enc_onnx),
        input_names=['src', 'src_mask'],
        output_names=['encoder_output'],
        dynamic_axes={
            'src': {0: 'batch', 1: 'seq'},
            'src_mask': {0: 'batch', 2: 'seq', 3: 'seq'},
            'encoder_output': {0: 'batch', 1: 'seq'}
        }
    )

    build_single_engine(
        str(enc_onnx), str(enc_engine),
        min_shapes="src:1x1,src_mask:1x1x1x1",
        opt_shapes=f"src:1x{seq_len},src_mask:1x1x{seq_len}x{seq_len}",
        max_shapes=f"src:8x{seq_len},src_mask:8x1x{seq_len}x{seq_len}",
        use_fp16 = USE_FP16 # Forcing the encoder to be fp32 due to accuracy 
    )

    # ==========================================================================
    # DECODER
    # ==========================================================================
    dec_wrapper = DecoderWrapper(model)
    dec_onnx = onnx_dir / f"{model_base_name}_decoder.onnx"
    dec_engine = trt_dir / f"{model_base_name}_decoder.engine"

    # Encoder output dummy (full src length is fine)
    dummy_enc_out = torch.randn(bs, seq_len, d_model)

    # export decoder for incremental decode
    dummy_tgt_len = 2   # 1 or 2 is enough to keep shapes symbolic
    dummy_tgt = torch.randint(0, vars["TGT_VOCAB_SIZE"], (bs, dummy_tgt_len))

    # causal lower-tri mask like in dataset.py file
    dummy_tgt_mask = torch.tril(torch.ones(bs, 1, dummy_tgt_len, dummy_tgt_len))

    # src_mask for cross-attn (binary, broadcast shape)
    dummy_src_mask = torch.ones(bs, 1, 1, seq_len)

    export_single_onnx(
        dec_wrapper,
        (dummy_enc_out, dummy_src_mask, dummy_tgt, dummy_tgt_mask),
        str(dec_onnx),
        input_names=['encoder_output', 'src_mask', 'tgt', 'tgt_mask'],
        output_names=['decoder_output'],
        dynamic_axes={
            'encoder_output': {0: 'batch', 1: 'src_seq'},
            'src_mask': {0: 'batch', 3: 'src_seq'},
            'tgt': {0: 'batch', 1: 'tgt_seq'},
            'tgt_mask': {0: 'batch', 2: 'tgt_seq', 3: 'tgt_seq'},
            'decoder_output': {0: 'batch', 1: 'tgt_seq'}
        }
        )

    build_single_engine(
        str(dec_onnx), str(dec_engine),
        min_shapes="encoder_output:1x1x512,src_mask:1x1x1x1,tgt:1x1,tgt_mask:1x1x1x1",
        opt_shapes=f"encoder_output:1x{seq_len}x{d_model},src_mask:1x1x1x{seq_len},tgt:1x{seq_len},tgt_mask:1x1x{seq_len}x{seq_len}",
        max_shapes=f"encoder_output:8x{seq_len}x{d_model},src_mask:8x1x1x{seq_len},tgt:8x{seq_len},tgt_mask:8x1x{seq_len}x{seq_len}",
        use_fp16=USE_FP16
    )

    # ==========================================================================
    # PROJECTION
    # ==========================================================================
    proj_wrapper = ProjectionWrapper(model)
    proj_onnx = onnx_dir / f"{model_base_name}_projection.onnx"
    proj_engine = trt_dir / f"{model_base_name}_projection.engine"

    dummy_dec_out = torch.randn(bs, seq_len, d_model)

    export_single_onnx(
        proj_wrapper,
        (dummy_dec_out,),
        str(proj_onnx),
        input_names=['decoder_output'],
        output_names=['logits'],
        dynamic_axes={
            'decoder_output': {0: 'batch', 1: 'seq'},
            'logits': {0: 'batch', 1: 'seq'}
        }
    )

    build_single_engine(
        str(proj_onnx), str(proj_engine),
        min_shapes=f"decoder_output:1x1x{d_model}",
        opt_shapes=f"decoder_output:1x{seq_len}x{d_model}",
        max_shapes=f"decoder_output:8x{seq_len}x{d_model}",
        use_fp16 = USE_FP16
    )

    print("\n🎉 All components exported successfully!")
    print(f"Files located in {onnx_dir} and {trt_dir}")

if __name__ == "__main__":
    main()