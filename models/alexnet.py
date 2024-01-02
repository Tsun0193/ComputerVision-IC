""" AlexNet model """
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet class"""

    def __init__(self, output_dim):
        """Constructor of class

        Args:
            output_dim (int): Number of classes for classification head.
        """
        super().__init__()
        # 5 convolutional layer, after each is ReLU activation function, MaxPooling layer is after Conv layer 1, 2, 5
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(6),
        )
        # 3 fully-connected layers 4096, 4096, 1000
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        """Forward function

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            A output tensor.
        """

        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x
