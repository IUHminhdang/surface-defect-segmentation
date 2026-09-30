"""ResNet34 feature encoder used by the segmentation models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet34_Weights, resnet34


class ResNet34Encoder(nn.Module):
	def __init__(self, pretrained: bool = False) -> None:
		super().__init__()
		weights = ResNet34_Weights.DEFAULT if pretrained else None
		backbone = resnet34(weights=weights)
		self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
		self.pool = backbone.maxpool
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		self.layer3 = backbone.layer3
		self.layer4 = backbone.layer4

	def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		stem = self.stem(inputs)
		feature1 = self.layer1(self.pool(stem))
		feature2 = self.layer2(feature1)
		feature3 = self.layer3(feature2)
		feature4 = self.layer4(feature3)
		return stem, feature1, feature2, feature3, feature4
