from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass(frozen=True)
class FIDStats:
    mean: torch.Tensor
    covariance: torch.Tensor
    num_samples: int


class InceptionFID(nn.Module):
    """FID feature extractor and statistic calculator."""

    def __init__(self, device: torch.device | str) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=None, aux_logits=True, transform_input=False)
        model.load_state_dict(weights.get_state_dict(progress=True))
        self.features = create_feature_extractor(model.eval(), return_nodes={"avgpool": "features"})
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.detach().float().clamp(-1.0, 1.0)
        images = (images + 1.0) / 2.0
        if images.shape[-2:] != (299, 299):
            images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        features = self.features(images)["features"]
        return features.flatten(1).cpu()


def calculate_stats(features: torch.Tensor) -> FIDStats:
    if features.ndim != 2:
        raise ValueError(f"expected feature matrix [N, D], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("FID requires at least two samples")

    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T.matmul(centered) / (features.shape[0] - 1)
    return FIDStats(mean=mean, covariance=covariance, num_samples=features.shape[0])


def calculate_fid(real: FIDStats, fake: FIDStats, eps: float = 1.0e-6) -> float:
    if real.mean.shape != fake.mean.shape:
        raise ValueError("real and fake FID feature dimensions do not match")

    diff = real.mean - fake.mean
    cov_real = real.covariance
    cov_fake = fake.covariance
    if not torch.isfinite(cov_real).all() or not torch.isfinite(cov_fake).all():
        raise ValueError("FID covariance contains non-finite values")

    eye = torch.eye(cov_real.shape[0], dtype=cov_real.dtype, device=cov_real.device)
    product = (cov_real + eps * eye).matmul(cov_fake + eps * eye)
    eigvals = torch.linalg.eigvals(product).real.clamp_min(0.0)
    trace_sqrt_product = eigvals.sqrt().sum()
    fid = diff.dot(diff) + torch.trace(cov_real) + torch.trace(cov_fake) - 2.0 * trace_sqrt_product
    return float(fid.clamp_min(0.0).item())


def stats_from_feature_batches(feature_batches: list[torch.Tensor]) -> FIDStats:
    if not feature_batches:
        raise ValueError("no feature batches were collected")
    return calculate_stats(torch.cat(feature_batches, dim=0))
