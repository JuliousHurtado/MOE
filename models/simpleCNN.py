import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network
    """
    def __init__(self, num_channels, N, num_classes=10, add_pooling=False):
        super(SimpleCNN, self).__init__()

        if add_pooling:
            stride=1
        else:
            stride=2

        layer = nn.Sequential()
        layer.add_module('conv1',nn.Conv2d(3, num_channels[0]*N, kernel_size=3, stride=stride))
        layer.add_module('relu1',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool1',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv2',nn.Conv2d(num_channels[0]*N, num_channels[1]*N, kernel_size=3, stride=stride))
        layer.add_module('relu2',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool2',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv3',nn.Conv2d(num_channels[1]*N, num_channels[2]*N, kernel_size=3, stride=stride))
        layer.add_module('relu3',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool3',nn.MaxPool2d(kernel_size=2, stride=1))
        self.features = layer

        self.classifier = nn.Sequential(nn.Linear(1152*N, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    N = 2
    x = torch.rand(10,3,32,32)
    model = SimpleCNN([32,64,128],4,add_pooling=False)
    print(model)
    model(x)