"""Defines a generic convolutional network."""
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Convolutional network."""

    def __init__(self, widen_factor=2):
        """Describe."""
        super().__init__()
        self.channels = [32, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.layers = self._make_conv_layers()
        self.classifier = self._make_classifier()        

    def forward(self, x):
        out = self.layers(x)
        out = out.view(-1, self.channels[2] * 2 * 2)
        out = self.classifier(out)
        return out

    def _make_conv_layers(self):
        return nn.Sequential(
            nn.Conv2d(3, self.channels[0], 5),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.channels[0], self.channels[1], 3),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.channels[1], self.channels[2], 3),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
    
    def _make_classifier(self):
        return nn.Sequential(
            nn.Linear(self.channels[2] * 2 * 2, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(100, 10))
       
