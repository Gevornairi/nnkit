"""Loss functions and their gradients."""

from .label_smoothing_loss import LabelSmoothingLoss
from .z_loss import z_loss

__all__ = ["LabelSmoothingLoss", "z_loss"]
