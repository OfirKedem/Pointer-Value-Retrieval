import torch
from torch import nn
from datasets import BlockStylePVR

from torch.utils.data import DataLoader
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        # get original model
        resnet18 = models.resnet18()

        # modify input channels
        input_layer = resnet18.conv1
        resnet18.conv1 = nn.Conv2d(1,
                                   input_layer.out_channels,
                                   kernel_size=input_layer.kernel_size,
                                   stride=input_layer.stride,
                                   padding=input_layer.padding,
                                   bias=input_layer.bias)

        # modify last layer to match output classes
        num_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_features, 10)

        self.model = resnet18

    def forward(self, x):
        output = self.model(x)
        return output


def main():
    ds = BlockStylePVR(train=True, size=10)
    net = ResNet18()
    dl = DataLoader(ds, batch_size=2)

    for sample, value in dl:
        output = net(sample)
        print(output.shape)

        break


if __name__ == '__main__':
    main()
