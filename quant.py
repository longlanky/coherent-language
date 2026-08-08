"""INT8 weight-only quantization for the Cohere ASR encoder.

Reproduces the FluidInference CoreML hybrid scheme natively in PyTorch:
per-output-channel symmetric INT8 encoder weights with FP16 activations,
while the decoder, embeddings, LayerNorms, and conv layers stay FP16.

The encoder holds the large majority of the model's parameters, so this
cuts weight memory roughly in half overall (~4x for the encoder itself),
which matters on the 4 GB Quadro P1000.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("cohere-asr.quant")


class Int8Linear(nn.Module):
    """Drop-in nn.Linear replacement with per-channel INT8 weights.

    Weights are stored as int8 plus one fp16 scale per output channel and
    dequantized to fp16 on the fly, so activations remain FP16 exactly as
    in the unquantized model.
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)  # [out, in] int8
        self.register_buffer("scale", scale)  # [out, 1] fp16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        w = linear.weight.detach()
        absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 127.0
        w_int8 = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(w_int8, scale.to(w.dtype), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.weight_int8.to(self.scale.dtype) * self.scale).to(x.dtype)
        return F.linear(x, weight, self.bias)


def quantize_encoder_int8(asr_model) -> int:
    """Replace every nn.Linear in the model's encoder with Int8Linear, in-place.

    Returns the number of quantized layers.
    """
    encoder = asr_model.model.encoder
    before = sum(p.numel() * p.element_size() for p in encoder.parameters())

    count = 0
    for module in encoder.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                with torch.no_grad():
                    setattr(module, name, Int8Linear.from_linear(child))
                count += 1

    after = sum(p.numel() * p.element_size() for p in encoder.parameters())
    log.info(
        "Quantized %d encoder Linear layers to INT8: %.2f GB -> %.2f GB",
        count, before / 2**30, after / 2**30,
    )
    return count
