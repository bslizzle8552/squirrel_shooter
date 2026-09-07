#!/usr/bin/env python3
"""Export a static raw-head TorchScript intermediate for direct PNNX conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


EXPECTED_CHECKPOINT_SHA256 = "5d8b78244107302881477bda95353236f29356c23a2a10f61936225c4dfc4831"
EXPECTED_YOLOX_COMMIT = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3"


class FocusAsStrideTwoConv(nn.Module):
    """Exact YOLOX Focus slicing expressed with NCNN-supported convolution."""

    def __init__(self, focus: nn.Module, in_channels: int = 3) -> None:
        super().__init__()
        self.rearrange = nn.Conv2d(
            in_channels,
            in_channels * 4,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        with torch.no_grad():
            self.rearrange.weight.zero_()
            offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
            for group, (row, column) in enumerate(offsets):
                for channel in range(in_channels):
                    self.rearrange.weight[group * in_channels + channel, channel, row, column] = 1.0
        self.rearrange.weight.requires_grad_(False)
        self.conv = focus.conv

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.rearrange(inputs))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exp-file", type=Path, required=True)
    parser.add_argument("--yolox-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint identity mismatch")
    commit = subprocess.check_output(["git", "-C", str(args.yolox_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_YOLOX_COMMIT:
        raise RuntimeError("YOLOX source identity mismatch")
    sys.path.insert(0, str(args.yolox_source))
    from yolox.exp import get_exp  # noqa: PLC0415
    from yolox.models.network_blocks import Focus, SiLU  # noqa: PLC0415
    from yolox.utils import replace_module  # noqa: PLC0415

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    exp = get_exp(str(args.exp_file), None)
    model = exp.get_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    dummy = torch.zeros((1, 3, 416, 416), dtype=torch.float32)
    with torch.inference_mode():
        reference = model(dummy)
        focus_modules = [name for name, module in model.named_modules() if isinstance(module, Focus)]
        if focus_modules != ["backbone.backbone.stem"]:
            raise RuntimeError(f"Unexpected YOLOX Focus module locations: {focus_modules}")
        model.backbone.backbone.stem = FocusAsStrideTwoConv(model.backbone.backbone.stem)
        rewritten = model(dummy)
        rewrite_delta = float(torch.max(torch.abs(reference - rewritten)).item())
        if rewrite_delta > 0.00005:
            raise RuntimeError(f"Focus rewrite changed eager output: {rewrite_delta}")
        traced = torch.jit.trace(model, dummy, strict=True)
        traced = torch.jit.freeze(traced.eval())
        traced_output = traced(dummy)
    maximum_delta = float(torch.max(torch.abs(reference - traced_output)).item())
    if tuple(traced_output.shape) != (1, 3549, 6) or maximum_delta > 0.00005:
        raise RuntimeError(f"TorchScript trace mismatch: shape={tuple(traced_output.shape)}, delta={maximum_delta}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    metadata = {
        "status": "COMPLETE",
        "purpose": "Direct PNNX conversion intermediate; not a Pi runtime artifact.",
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "input_shape": [1, 3, 416, 416],
        "output_shape": [1, 3549, 6],
        "decoded_in_graph": False,
        "fp32": True,
        "maximum_abs_delta_vs_eager_zero_input": maximum_delta,
        "focus_export_rewrite": "Exact one-hot 2x2 stride-2 convolution replacing slice/concat only in the derived export graph.",
        "focus_rewrite_maximum_abs_delta_vs_original_zero_input": rewrite_delta,
        "torch": torch.__version__,
        "yolox_commit": commit,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
