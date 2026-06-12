from __future__ import annotations

import torch

from src.macro.dit import build_dit_flow
from src.macro.unet import build_unet_flow


def build_macro_flow_model(config: dict[str, object]) -> torch.nn.Module:
    architecture = str(config.get("architecture", "unet")).lower()
    if architecture in {"unet", "unet_flow"}:
        return build_unet_flow(config)
    if architecture in {"dit", "dit-b", "dit-b/2", "dit-b/4"}:
        merged_config = dict(config)
        if architecture != "dit" and "variant" not in merged_config:
            merged_config["variant"] = architecture
        return build_dit_flow(merged_config)
    raise ValueError(f"unsupported macro flow architecture: {architecture}")
