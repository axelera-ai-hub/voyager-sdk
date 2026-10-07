# Copyright Axelera AI, 2025
import logging
from typing import Any, Dict

import torch.nn as nn

LOG = logging.getLogger(__name__)


def convert_first_node_to_1_channel(model, conversion_method='sum'):
    """
    Convert the first convolutional layer to accept grayscale input (1 channel).

    Args:
        model: The PyTorch model to modify
        conversion_method: How to combine RGB channels - options:
            - 'sum': Sum the weights of all channels
            - 'weighted': Use RGB to grayscale conversion weights (0.2989, 0.5870, 0.1140)
            - 'average': Simple average of the channel weights

    Returns:
        Modified model with first convolutional layer accepting 1-channel input
    """

    # Find the first convolutional layer in the model
    first_conv = None
    first_conv_name = None

    LOG.info(
        f"Input requires 1 channel, model expects 3 channels. Converting first layer using {conversion_method} method"
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            first_conv = module
            first_conv_name = name
            break

    if first_conv is None:
        raise ValueError("Could not find a convolutional layer with 3 input channels")

    # Get original weights
    original_weights = first_conv.weight.data

    # Apply the specified conversion method
    if conversion_method == 'sum':
        new_weights = original_weights.sum(dim=1, keepdim=True)
    elif conversion_method == 'weighted':
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=original_weights.device).view(
            1, 3, 1, 1
        )
        new_weights = (original_weights * rgb_weights).sum(dim=1, keepdim=True)
    elif conversion_method == 'average':
        new_weights = original_weights.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown conversion method: {conversion_method}")

    # Create a new conv layer with same parameters but 1 input channel
    new_conv = nn.Conv2d(
        1,
        out_channels=original_weights.size(0),
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
        dilation=first_conv.dilation,
        groups=1,
        padding_mode=first_conv.padding_mode,
    )

    # Set the weights of the new layer
    new_conv.weight = nn.Parameter(new_weights)

    # If the original layer had bias, copy it
    if first_conv.bias is not None:
        new_conv.bias = nn.Parameter(first_conv.bias.data.clone())

    # Replace the original layer with the new one
    parts = first_conv_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

    LOG.warning(
        "NOTE: Based on experiments, duplicating grayscale to 3 channels in preprocessing "
        "can yield better accuracy than modifying the model architecture. "
        "Consider using a 3-channel input with duplicated grayscale values."
    )

    return model
