"""Attention gate for filtering encoder skip features."""

from __future__ import annotations

from torch import Tensor, nn


class AttentionGate(nn.Module):
	def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
		super().__init__()
		self.gate = nn.Sequential(
			nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(inter_channels),
		)
		self.skip = nn.Sequential(
			nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(inter_channels),
		)
		self.score = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(inter_channels, 1, kernel_size=1),
			nn.Sigmoid(),
		)

	def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
		attention = self.score(self.gate(gate) + self.skip(skip))
		return skip * attention
