from __future__ import annotations

import copy

import torch


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self.shadow[name].device != parameter.device or self.shadow[name].dtype != parameter.dtype:
                self.shadow[name] = self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow = copy.deepcopy(state_dict["shadow"])

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
