import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    """
    Neural Network
    """
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()

        layer = nn.Sequential()
        layer.add_module('flatten', nn.Flatten())
        layer.add_module('linear1', nn.Linear(784, 512))
        layer.add_module('relu1',nn.ReLU(inplace=True))
        layer.add_module('linear2', nn.Linear(512, 256))
        layer.add_module('relu2',nn.ReLU(inplace=True))

        self.features = layer

        self.classifier = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

