import math
from collections import OrderedDict

import torch
import torch.nn as nn

from model import base_model


# -----------------------------------------------------------------------------------------------------------#
# class DarkNet(nn.Module): # DarkNet 网络结构
# def darknet53(pretrained): # DarkNet53
# -----------------------------------------------------------------------------------------------------------#


class DarkNet(nn.Module):
    """
    DarkNet 网络结构

    其特征层为：
    208,208,64
    104,104,128
    52,52,256
    26,26,512
    13,13,1024
    """

    def __init__(self, block_counts: list) -> None:
        """
        :param block_counts: 每个残差层的重复数目
        """
        super().__init__()

        self.input_channels = 3  # 输入图片通道数
        self.init_channels = 32  # 初始卷积通道数

        # 初始卷积
        self.init_conv = base_model.Conv2d(self.input_channels, self.init_channels, 3)  # 416,416,3 -> 416,416,32

        # 五层下采样加残差卷积
        self.layer1 = self._make_layer(32, 64, 32, block_counts[0])  # 416,416,32 -> 208,208,64
        self.layer2 = self._make_layer(64, 128, 64, block_counts[1])  # 208,208,64 -> 104,104,128
        self.layer3 = self._make_layer(128, 256, 128, block_counts[2])  # 104,104,128 -> 52,52,256
        self.layer4 = self._make_layer(256, 512, 256, block_counts[3])  # 52,52,256 -> 26,26,512
        self.layer5 = self._make_layer(512, 1024, 512, block_counts[4])  # 26,26,512 -> 13,13,1024

        # DarkNet53 的五个特征层的通道数
        self.layers_output_channels = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():  # 遍历所有模型（所有层级的，但是只需要对最基础层级的模型进行权值初始化）
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 卷积核的参数数目
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 简化的 kaiming 高斯初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # 初始化权重为 1
                m.bias.data.zero_()  # 初始化偏置为 0

    # -----------------------------------------------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为 2 的 3x3 卷积进行下采样，然后进行残差结构的重复堆叠
    # -----------------------------------------------------------------------------------------------------------#
    def _make_layer(self,
                    input_channels: int,
                    block_channels: int, block_inner_channels: int,
                    block_count: int) -> nn.Module:
        """
        :param input_channels: 初始输入通道数
        :param block_channels: 残差结构的输入和输出通道数，因为残差结构需要重复堆叠，所以残差结构的输入通道数应该等于残差结构的输出通道数
        :param block_inner_channels: 残差结构的中间通道数
        :param block_count: 残差结构的数目
        :return:
        """
        layers = []
        # 1. 下采样
        layers.append(
            (
                "down_sample_conv",
                base_model.Conv2d(input_channels, block_channels, 3, 2)
            )
        )
        # 2. 堆叠残差结构
        for i in range(0, block_count):
            layers.append(
                (
                    "residual_{}".format(i),
                    base_model.BasicBlock(block_channels, block_inner_channels, block_channels)
                )
            )
        # 3. 堆叠所有层
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.init_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)

        feature_52 = self.layer3(x)
        feature_26 = self.layer4(feature_52)
        feature_13 = self.layer5(feature_26)

        return feature_52, feature_26, feature_13


def darknet53(pretrained=False):
    """
    DarkNet53
    DarkNet53 一共 53 层卷积，除去最后一个 FC（全连接层，实际上是通过 1x1 卷积实现的）总共 52 个卷积，作为特征提取网络

    :param pretrained: 是否预训练，若预训练，传入权重的存储路径，默认为 False
    :return:
    """
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet53 except a pretrained path. got [{}]".format(pretrained))
    return model


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    backbone = darknet53()
    print(backbone)

    """
DarkNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): LeakyReLU(negative_slope=0.1)
  (layer1): Sequential(
    (ds_conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (ds_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (ds_relu): LeakyReLU(negative_slope=0.1)
    (residual_0): BasicBlock(
      (conv1): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
  )
  (layer2): Sequential(
    (ds_conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (ds_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (ds_relu): LeakyReLU(negative_slope=0.1)
    (residual_0): BasicBlock(
      (conv1): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_1): BasicBlock(
      (conv1): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
  )
  (layer3): Sequential(
    (ds_conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (ds_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (ds_relu): LeakyReLU(negative_slope=0.1)
    (residual_0): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_1): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_2): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_3): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_4): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_5): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_6): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_7): BasicBlock(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
  )
  (layer4): Sequential(
    (ds_conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (ds_bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (ds_relu): LeakyReLU(negative_slope=0.1)
    (residual_0): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_1): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_2): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_3): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_4): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_5): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_6): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_7): BasicBlock(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
  )
  (layer5): Sequential(
    (ds_conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (ds_bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (ds_relu): LeakyReLU(negative_slope=0.1)
    (residual_0): BasicBlock(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_1): BasicBlock(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_2): BasicBlock(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
    (residual_3): BasicBlock(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): LeakyReLU(negative_slope=0.1)
      (conv2): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): LeakyReLU(negative_slope=0.1)
    )
  )
)
    """
