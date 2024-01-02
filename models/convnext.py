""" ConvNeXt model """
""" Reference: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class Block(nn.Module):
    """Block class"""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        """Contructor of class

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic drop path rate. Default 0.0.
            layer_scale_init_value (float): Init value for Layer Scale. Default 1e-6.
        """

        super().__init__()
        # Depwise convolutional layer
        self.dwconv = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        # Pointwise convolutional layer
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward funtion

        Args:
            x (torch.Tensor): input tensor.
        Returns:
            A output tensor.
        """

        identity = x.clone()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = identity + self.drop_path(x)

        return x


class ConvNeXt(nn.Module):
    """ConvNeXt class"""

    def __init__(
        self,
        num_classes=1000,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        """Constructor of the class

        Args:
            num_classes (int): Number of classes for classification head. Default: 1000.
            in_chans (int): Number of input image channels. Default: 3.
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3].
            dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768].
            drop_path_rate (float): Stochastic drop path rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """

        super().__init__()
        # Stem and 3 intermediate downsampling convolutional layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final norm layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        """Forward function

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            A output tensor.
        """

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(
            x.mean([-2, -1])
        )  # Global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)

        return x


class LayerNorm(nn.Module):
    """LayerNorm class"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """Constructor of the class

        Args:
            normalized_shape (tuple(int, int)): input shape from expected input of size.
            eps (float): a value added to the denominator for numerical stability. Default: 1e-6.
            data_format: one of two options [channels_last, channel_first], describe how normalized tensor construct
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            A normalized tensor.
        """

        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
