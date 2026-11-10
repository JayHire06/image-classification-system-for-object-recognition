"""MobileNetV2 model construction and transfer-learning controls."""

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNetV2

from .dataset import CLASS_NAMES

PRETRAINED_WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V2


def build_mobilenet_v2(*, pretrained: bool = True) -> MobileNetV2:
    """Build a MobileNetV2 with a CIFAR-10 classification head.

    Args:
        pretrained: Load the pinned ImageNet weights for transfer learning.
            Use False when restoring a project checkpoint.

    Returns:
        A MobileNetV2 with ten output logits in CIFAR-10 class order.
    """
    model = mobilenet_v2(weights=PRETRAINED_WEIGHTS if pretrained else None)
    model.classifier[-1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    return model


def set_trainable_layers(model: MobileNetV2, top_blocks: int) -> None:
    """Enable gradients for the classifier and a suffix of the backbone.

    Args:
        model: The MobileNetV2 to configure.
        top_blocks: Number of final backbone blocks to fine-tune. Use zero
            for head-only training.

    Raises:
        ValueError: If the requested block count is outside the backbone.
    """
    if not 0 <= top_blocks <= len(model.features):
        raise ValueError("The fine-tuning block count exceeds the backbone.")
    model.features.requires_grad_(False)
    for index in range(len(model.features) - top_blocks, len(model.features)):
        model.features[index].requires_grad_(True)
    model.classifier.requires_grad_(True)


def set_training_mode(model: MobileNetV2, top_blocks: int) -> None:
    """Train the active layers while keeping frozen batch statistics fixed.

    Args:
        model: The MobileNetV2 to put in training mode.
        top_blocks: Number of unfrozen final backbone blocks.
    """
    model.train()
    for index in range(len(model.features) - top_blocks):
        model.features[index].eval()
