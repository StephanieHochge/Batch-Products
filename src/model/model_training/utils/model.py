from torchvision import models
import torch
from torch import nn


def get_model(device: torch.device, num_classes: int = 10,
              dropout_rate: float = 0.0, freeze: bool = False) -> nn.Module:
    """
    Initialize and return an EfficientNet-B0 model adapted for grayscale images.

    :param device: The device to perform computation on (e.g., "cpu" or "cuda")
    :param num_classes: The number of output classes for the classifier. Default is 10.
    :param dropout_rate: The dropout rate for the classifier head. Default is 0.0.
    :param freeze: Whether to freeze all layers except the classifier. Default is False.
    :return: The modified EfficientNet-B0 model.
    """
    # load a pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0()

    # modify the first convolution layer to accept 1 input channel (for grayscale images)
    old_conv = model.features[0][0]

    # create a new convolutional layer with 1 input channel instead of 3
    new_conv = nn.Conv2d(
        in_channels=1,  # grayscale has 1 channel
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # copy existing weights for the first channel and average over the 3 input channels
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

    # replace the first layer with the new convolutional layer
    model.features[0][0] = new_conv

    # optionally freeze layers except for the classifier head
    if freeze:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # adjust the classifier for the number of target classes
    model.classifier[1] = nn.Sequential(
        nn.Dropout(dropout_rate),  # add dropout
        nn.Linear(model.classifier[1].in_features, num_classes)  # adjust the output features
    )

    # move the model to the specified device
    return model.to(device)
