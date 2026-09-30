"""Multi-scale feature fusion block."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiScaleFeatureFusion(nn.Module):
	def __init__(self, channels: int) -> None:
		super().__init__()
		branch_channels = channels // 4
		self.branches = nn.ModuleList(
			[
				nn.Conv2d(channels, branch_channels, kernel_size=1),
				nn.Conv2d(channels, branch_channels, kernel_size=3, padding=1),
				nn.Conv2d(channels, branch_channels, kernel_size=3, padding=2, dilation=2),
				nn.Conv2d(channels, branch_channels, kernel_size=3, padding=4, dilation=4),
			]
		)
		self.project = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, inputs: Tensor) -> Tensor:
		fused = torch.cat([branch(inputs) for branch in self.branches], dim=1)
		return self.project(fused) + inputs
