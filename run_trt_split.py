import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import torch
import os

# --- JETSON SPECIFIC OPTIMIZATION ---
# Limit PyTorch GPU memory so TRT has headroom.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.6)
# ------------------------------------

# Initialize CUDA driver
cuda.init()

# Import your existing validation/config logic
from config import get_config
from train import get_ds
from dataset import causal_mask


class TRTEngine:
    """
    TensorRT 10.3+ wrapper using execute_async_v3, shared PRIMARY CUDA context
    (safe to mix with PyTorch on Jetson).
    """
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)

        # Ensure PyTorch initializes CUDA first (primary context)
        if torch.cuda.is_available():
            torch.cuda.init()

        # Retain PRIMARY context shared with PyTorch
        self.device = cuda.Device(0)
        self.cfx = self.device.retain_primary_context()

        # Create TRT objects/stream inside the correct context
        self.cfx.push()
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            # If multiple optimization profiles exist, pick profile 0
            nprof = getattr(self.engine, "num_optimization_profiles", 1)
            if nprof and nprof > 1:
                self.context.set_optimization_profile_async(0, self.stream.handle)

        finally:
            self.cfx.pop()

    def numpy_dtype_for(self, name: str):
        return trt.nptype(self.engine.get_tensor_dtype(name))

    def infer(self, inputs_dict):
        """
        inputs_dict: { "tensor_name": numpy_array }
        returns: { "tensor_name": host_numpy_array }
        """
        self.cfx.push()
        try:
            device_allocations = []
            outputs = {}

            # 1. Set Input Shapes & Allocate Input Memory
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)

                if mode != trt.TensorIOMode.INPUT:
                    continue

                if name not in inputs_dict:
                    # enqueueV3 requires all inputs to be bound
                    raise RuntimeError(f"Missing required input tensor '{name}' for engine.")

                data = inputs_dict[name]

                # TRT 10.X: Set dynamic shape for this input
                self.context.set_input_shape(name, data.shape)

                # Allocate GPU memory for input
                d_input = cuda.mem_alloc(data.nbytes)
                cuda.memcpy_htod_async(d_input, data, self.stream)

                # TRT 10.X: Bind tensor address
                self.context.set_tensor_address(name, int(d_input))
                device_allocations.append(d_input)

            # 2. Resolve Output Shapes & Allocate Output Memory
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)

                if mode != trt.TensorIOMode.OUTPUT:
                    continue

                dims = tuple(self.context.get_tensor_shape(name))

                if any(d < 0 for d in dims):
                    raise RuntimeError(
                        f"Unresolved output shape for '{name}': {dims}. "
                        f"This usually means an input shape/mask mismatch."
                    )

                dtype = self.numpy_dtype_for(name)

                # Host (CPU) + Device (GPU)
                h_output = cuda.pagelocked_empty(dims, dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)

                self.context.set_tensor_address(name, int(d_output))

                outputs[name] = (h_output, d_output)
                device_allocations.append(d_output)

            # 3. Execute (V3)
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False.")

            # 4. Copy Outputs Back
            final_outputs = {}
            for name, (h_out, d_out) in outputs.items():
                cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
                final_outputs[name] = h_out

            self.stream.synchronize()
            return final_outputs

        finally:
            self.cfx.pop()


class TRTTransformer:
    def __init__(self, enc_path, dec_path, proj_path):
        print("Loading TRT Engines...")
        self.encoder = TRTEngine(enc_path)
        self.decoder = TRTEngine(dec_path)
        self.projector = TRTEngine(proj_path)
        print("Engines loaded.")

    def to_numpy_for(self, trt_engine: TRTEngine, name: str, tensor: torch.Tensor):
        np_dtype = trt_engine.numpy_dtype_for(name)
        return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np_dtype))

    def encode(self, src, src_mask):
        # ENCODER expects FLOAT binary square mask (B, 1, S, S)

        seq_len = src.shape[1]

        # src_mask from ds is (B,1,1,S) int {0,1}
        if src_mask.dim() == 3:
            src_mask = src_mask.unsqueeze(1)  # (B,1,S)->(B,1,1,S)

        # force square for encoder
        if src_mask.dim() == 4 and src_mask.shape[2] == 1:
            src_mask = src_mask.repeat(1, 1, seq_len, 1)  # (B,1,1,S)->(B,1,S,S)

        # keep it binary, just cast to float 
        src_mask = src_mask.float()

        inputs = {
            "src": self.to_numpy_for(self.encoder, "src", src),                 # INT64
            "src_mask": self.to_numpy_for(self.encoder, "src_mask", src_mask), # FLOAT binary
        }

        out = self.encoder.infer(inputs)
        return torch.from_numpy(out["encoder_output"]).cuda()

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # DECODER expects FLOAT binary masks:
        #   src_mask: (B,1,1,S)
        #   tgt_mask: (B,1,T,T)

        if src_mask.dim() == 3:
            src_mask = src_mask.unsqueeze(1)

        if tgt_mask.dim() == 3:
            tgt_mask = tgt_mask.unsqueeze(1)

        # keep binary, cast to float
        src_mask = src_mask.float()
        tgt_mask = tgt_mask.float()

        inputs = {
            "encoder_output": self.to_numpy_for(self.decoder, "encoder_output", encoder_output),  # FLOAT
            "src_mask": self.to_numpy_for(self.decoder, "src_mask", src_mask),                   # FLOAT binary
            "tgt": self.to_numpy_for(self.decoder, "tgt", tgt),                                  # INT64
            "tgt_mask": self.to_numpy_for(self.decoder, "tgt_mask", tgt_mask),                   # FLOAT binary
        }

        out = self.decoder.infer(inputs)
        return torch.from_numpy(out["decoder_output"]).cuda()

    def project(self, x):
        is_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            is_2d = True

        inputs = {
            "decoder_output": self.to_numpy_for(self.projector, "decoder_output", x)
        }
        out = self.projector.infer(inputs)
        res = torch.from_numpy(out["logits"]).cuda()

        if is_2d:
            res = res.squeeze(1)
        return res


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_mask = decoder_mask.unsqueeze(1)  # (1,T,T)->(1,1,T,T)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input,
             torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation_trt(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device):
    count = 0
    print("Starting TRT Inference Validation...")

    device = torch.device("cuda")

    for batch in validation_ds:
        count += 1
        encoder_input = batch['encoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)

        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]

        model_out = greedy_decode(
            model, encoder_input, encoder_mask,
            tokenizer_src, tokenizer_tgt, max_len, device
        )

        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        print('-' * 80)
        print(f"SOURCE:    {source_text}")
        print(f"TARGET:    {target_text}")
        print(f"TRT PRED:  {model_out_text}")

        if count == 2:
            break


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    trt_model = TRTTransformer(
        enc_path="tensorrt_split/tmodel_10_encoder.engine",
        dec_path="tensorrt_split/tmodel_10_decoder.engine",
        proj_path="tensorrt_split/tmodel_10_projection.engine"
    )

    run_validation_trt(
        trt_model,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
        config['seq_len'],
        torch.device("cuda")
    )

