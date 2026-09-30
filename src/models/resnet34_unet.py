"""Baseline ResNet34-UNet model (experiment E0)."""

from __future__ import annotations

from torch import Tensor, nn

from .decoder import UNetDecoder
from .multiscale import MultiScaleFeatureFusion
from .resnet_encoder import ResNet34Encoder


class ResNet34UNet(nn.Module):
	def __init__(
		self,
		pretrained: bool = False,
		multi_scale: bool = False,
		attention: bool = False,
	) -> None:
		super().__init__()
		self.encoder = ResNet34Encoder(pretrained=pretrained)
		self.multi_scale = MultiScaleFeatureFusion(512) if multi_scale else nn.Identity()
		self.decoder = UNetDecoder(attention=attention)

	def forward(self, inputs: Tensor) -> Tensor:
		features = list(self.encoder(inputs))
		features[-1] = self.multi_scale(features[-1])
		return self.decoder(*features)
