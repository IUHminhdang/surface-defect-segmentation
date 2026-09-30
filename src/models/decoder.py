"""U-Net decoder blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention_gate import AttentionGate


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, inputs: Tensor) -> Tensor:
		return self.block(inputs)


class UpBlock(nn.Module):
	def __init__(
		self,
		in_channels: int,
		skip_channels: int,
		out_channels: int,
		attention: bool = False,
	) -> None:
		super().__init__()
		self.attention = (
			AttentionGate(in_channels, skip_channels, max(1, min(in_channels, skip_channels) // 2))
			if attention
			else None
		)
		self.block = ConvBlock(in_channels + skip_channels, out_channels)

	def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
		inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		if self.attention is not None:
			skip = self.attention(inputs, skip)
		return self.block(torch.cat((inputs, skip), dim=1))


class UNetDecoder(nn.Module):
	def __init__(self, attention: bool = False) -> None:
		super().__init__()
		self.up3 = UpBlock(512, 256, 256, attention)
		self.up2 = UpBlock(256, 128, 128, attention)
		self.up1 = UpBlock(128, 64, 64, attention)
		self.up0 = UpBlock(64, 64, 64, attention)
		self.head = nn.Conv2d(64, 1, kernel_size=1)

	def forward(
		self,
		stem: Tensor,
		feature1: Tensor,
		feature2: Tensor,
		feature3: Tensor,
		feature4: Tensor,
	) -> Tensor:
		decoded = self.up3(feature4, feature3)
		decoded = self.up2(decoded, feature2)
		decoded = self.up1(decoded, feature1)
		decoded = self.up0(decoded, stem)
		decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
		return self.head(decoded)
