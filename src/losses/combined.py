"""Combined losses used by the experiments."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from .dice import dice_loss
from .boundary import boundary_loss


def bce_dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
	bce = F.binary_cross_entropy_with_logits(logits, targets.float())
	return bce + dice_loss(logits, targets)


def combined_loss(
	logits: Tensor,
	targets: Tensor,
	use_boundary: bool = False,
	boundary_weight: float = 0.1,
) -> Tensor:
	loss = bce_dice_loss(logits, targets)
	if use_boundary:
		loss = loss + boundary_weight * boundary_loss(logits, targets)
	return loss
