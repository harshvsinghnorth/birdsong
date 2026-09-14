"""
A small CNN for log-mel spectrograms.

Deliberately simple: four conv blocks, global pooling, one linear layer.
About 400k parameters. Every piece has a stated reason, and the whole thing
fits on a whiteboard. A pretrained ImageNet backbone (EfficientNet, ResNet)
will probably score higher and is a later ablation -- "does pretraining on
photographs help on spectrograms?" -- not the starting point.

Input:  (batch, 1, N_MELS, frames)   float in [0, 1]
Output: (batch, n_classes)           raw logits. Apply sigmoid at predict
                                     time; BCEWithLogitsLoss does it during
                                     training in a numerically safe way.
"""

import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    """Conv -> BN -> ReLU -> MaxPool(2).

    3x3 kernels see local time-frequency texture (a harmonic edge, a sweep).
    BatchNorm keeps activations well-scaled layer to layer, which is most of
    what makes a from-scratch CNN train without learning-rate surgery.
    MaxPool halves both axes: after four blocks the (128, 313) input is
    (8, 19) -- coarse frequency region x coarse time region.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BirdCNN(nn.Module):
    def __init__(self, n_classes, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks.append(conv_block(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*blocks)

        # Global average pooling: collapse (channels, 8, 19) -> (channels,).
        # Asks "is each learned pattern present ANYWHERE in the window?" --
        # a call at second 1 and the same call at second 4 give the same
        # answer. A flatten would tie position to weights and break that.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Dropout on the 256-d summary vector is the one regulariser here.
        # 4,500 recordings is small for a CNN; without it the model
        # memorises train windows within a few epochs.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_in, n_classes),
        )

    def forward(self, x):
        x = self.features(x)          # (B, 256, 8, 19)
        x = self.pool(x).flatten(1)   # (B, 256)
        return self.head(x)           # (B, n_classes) logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
