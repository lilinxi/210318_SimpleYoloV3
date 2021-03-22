import torch
import torch.nn as nn


# -----------------------------------------------------------------------------------------------------------#
# class Conv2d(nn.Module) # 卷积网络结构
# class BasicBlock(nn.Module) # 残差块网络结构
# -----------------------------------------------------------------------------------------------------------#

class Conv2d(nn.Module):
    """
    卷积网络结构

    1. 卷积
    2. 批次标准化
    3. leaky relu 激活
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int = 1) -> None:
        """
        :param input_channels: 输入通道数
        :param output_channels: 输出通道数
        :param kernel_size: 卷积核大小，1 或 3
        :param stride: 步长，默认为 1
        """
        super().__init__()

        padding = (kernel_size - 1) // 2 if kernel_size else 0  # 根据卷积核的大小来计算填充的大小

        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 卷积
        out = self.conv(x)
        # 2. 批次标准化
        out = self.bn(out)
        # 3. leaky relu 激活
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    """
    残差块网络结构

    1. 利用一个 1x1 卷积提取特征并且下降通道数
    2. 利用一个 3x3 卷积提取特征并且上升通道数
    3. 加上残差
    """

    def __init__(self, input_channels: int, inner_channels: int, output_channels: int) -> None:
        """
        :param input_channels: 输入通道数
        :param inner_channels: 中间通道数
        :param output_channels: 输出通道数
        """
        super().__init__()

        # (batch_size, input_channels, height, width) -> (batch_size, inner_channels, height, width)
        self.conv1 = Conv2d(input_channels, inner_channels, 1)

        # (batch_size, inner_channels, height, width) -> (batch_size, output_channels, height, width)
        self.conv2 = Conv2d(inner_channels, output_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1. 利用一个 1x1 卷积提取特征并且下降通道数
        out = self.conv1(x)
        # 2. 利用一个 3x3 卷积提取特征并且上升通道数
        out = self.conv2(out)
        # 3. 加上残差
        out += residual

        return out


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    conv2d = Conv2d(16, 32, 3)
    print(conv2d)

    basisBlock = BasicBlock(32, 16, 32)
    print(basisBlock)
