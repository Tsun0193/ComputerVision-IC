""" Resnet model """
""" Reference: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/"""
import torch.nn as nn


class Block(nn.Module):
    """Block class"""

    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        """Constructor of the class

        Args:
            in_channels (int): Number of input image channel.
            out_channels (int): Number of classes for classification head.
            i_downsample (nn.Sequential, optional): a group of layers. Default: None.
            stride (int): the jump necessary to go from one element to the next one. Default: 1.
        """

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            A output tensor.
        """
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class ResNet152(nn.Module):
    """Resnet class"""

    def __init__(self, num_classes=1000, num_channels=3, layer_list=[3, 8, 36, 3]):
        """Constructor of the class

        Args:
            num_classes (int): Number of classes for classification head. Defaut: 1000.
            num_channels (int): Number of input image channels. Default: 3.
            layer_list (tuple(int)): Number of blocks at each stage. Default: [3, 8, 36, 3].
        """

        super().__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3)

        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(layer_list[0], planes=64)
        self.layer2 = self._make_layer(layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)

    def forward(self, x):
        """Forward function

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            A output tensor.
        """

        x = self.relu(self.batch_norm(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, blocks, planes, stride=1):
        """Make layer function

        Args:
            blocks (int): number of blocks in a stage.
            planes (int): the number of input feature maps.
            stride (int): the jump necessary to go from one element to the next one. Default: 1.
        Returns:
            A group of layers (in a stage)
        """

        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * Block.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * Block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * Block.expansion),
            )

        layers.append(
            Block(self.in_channels, planes, i_downsample=ii_downsample, stride=stride)
        )
        self.in_channels = planes * Block.expansion

        for _ in range(1, blocks):
            layers.append(Block(self.in_channels, planes))

        return nn.Sequential(*layers)
