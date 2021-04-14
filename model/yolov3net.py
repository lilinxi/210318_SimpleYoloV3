from typing import List

import torch
import torch.nn as nn

from model import base_model, darknet


# -----------------------------------------------------------------------------------------------------------#
# class YoloV3Net(nn.Module) # YoloV3 网络结构
# -----------------------------------------------------------------------------------------------------------#


class YoloV3Net(nn.Module):
    """
    YoloV3 网络结构
    DarkNet53 提取了三个有效特征层
    YoloV3 整合了三个最终预测层

    其有效特征层为：
    52,52,256
    26,26,512
    13,13,1024

    其最终预测层为：
    52,52,3*(4+1+cls)
    26,26,3*(4+1+cls)
    13,13,3*(4+1+cls)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config
        self.backbone = darknet.darknet53(False)

        # 最终预测层的通道数
        self.predict_output_channels_13 = len(config["anchors"][0]) * (5 + config["classes"])
        self.predict_output_channels_26 = len(config["anchors"][1]) * (5 + config["classes"])
        self.predict_output_channels_52 = len(config["anchors"][2]) * (5 + config["classes"])

        # channels: 1024 -> 512 -> 255
        self.last_layer_13 = self._make_predict_layer(
            self.backbone.layers_output_channels[-1],
            [512, 1024],
            self.predict_output_channels_13)

        # 上一层的中间预测结果进行特征整合，上采样：512*13*13 -> 256*26*26
        self.last_layer_26_conv = base_model.Conv2d(512, 256, 1)
        self.last_layer_26_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # channels: 512/2 + 512 -> 256 -> 255
        self.last_layer_26 = self._make_predict_layer(
            self.backbone.layers_output_channels[-2] + 256,
            [256, 512],
            self.predict_output_channels_26)

        # 上一层的中间预测结果进行特征整合，上采样：256*26*26 -> 128*52*52
        self.last_layer_52_conv = base_model.Conv2d(256, 128, 1)
        self.last_layer_52_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # channels: 256/2 + 256 -> 128 -> 255
        self.last_layer_52 = self._make_predict_layer(
            self.backbone.layers_output_channels[-3] + 128,
            [128, 256],
            self.predict_output_channels_52)

    def _make_predict_layer(self,
                            input_channels: int,
                            inner_channels_list: List[int],
                            output_channels: int
                            ) -> nn.ModuleList:
        """
        Yolo 的最终预测层，共有七层卷积网络，前五层用于提取特征，后两层用于获得 yolo 网络的预测结果
        :param input_channels: 输入通道数
        :param inner_channels_list: 中间通道数，[down_dimension_channels(特征整合通道数，即降维), feature_extract_channels(特征提取通道数)]
        :param output_channels: 输出通道数
        :return: 最终预测层的七层卷积网络
        """
        m = nn.ModuleList(
            [
                # 将输入降维
                base_model.Conv2d(input_channels, inner_channels_list[0], 1),

                base_model.Conv2d(inner_channels_list[0], inner_channels_list[1], 3),  # 特征提取
                base_model.Conv2d(inner_channels_list[1], inner_channels_list[0], 1),  # 特征整合
                base_model.Conv2d(inner_channels_list[0], inner_channels_list[1], 3),  # 特征提取
                base_model.Conv2d(inner_channels_list[1], inner_channels_list[0], 1),  # 特征整合
                base_model.Conv2d(inner_channels_list[0], inner_channels_list[1], 3),  # 特征提取

                # 降维到输出维度
                nn.Conv2d(inner_channels_list[1], output_channels, kernel_size=1, stride=1, padding=0, bias=True)
            ]
        )
        return m

    def _predict_layer_forward(self, input_layer: torch.Tensor, predict_layer: nn.ModuleList) \
            -> (torch.Tensor, torch.Tensor):
        """
        :param input_layer: 输入层
        :param predict_layer: 预测层
        :return: 预测值结果，中间上采样层
        """
        for i, layer in enumerate(predict_layer):
            input_layer = layer(input_layer)
            if i == 4:  # 前五层用于提取特征，后两层用于预测结果
                inner_branch = input_layer  # 提取特征
        return input_layer, inner_branch

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        :param x:
        :return: (batch_size, x+y+w+h+conf+classes, height, width)
        """
        # 三个有效特征层
        x_256_52, x_512_26, x_1024_13 = self.backbone(x)

        # 大预测层
        predict_13_feature, inner_predict_13 = self._predict_layer_forward(x_1024_13, self.last_layer_13)

        # 特征上移
        last_layer_26_in = self.last_layer_26_conv(inner_predict_13)  # 缩小通道数
        last_layer_26_in = self.last_layer_26_upsample(last_layer_26_in)  # 插值上采样
        last_layer_26_in = torch.cat([last_layer_26_in, x_512_26], 1)  # 拼接通道

        # 中预测层
        predict_26_feature, inner_predict_26 = self._predict_layer_forward(last_layer_26_in, self.last_layer_26)

        # 特征上移
        last_layer_52_in = self.last_layer_52_conv(inner_predict_26)  # 缩小通道数
        last_layer_52_in = self.last_layer_52_upsample(last_layer_52_in)  # 插值上采样
        last_layer_52_in = torch.cat([last_layer_52_in, x_256_52], 1)  # 拼接通道

        # 小预测层
        predict_52_feature, inner_predict_52 = self._predict_layer_forward(last_layer_52_in, self.last_layer_52)

        return predict_13_feature, predict_26_feature, predict_52_feature  # 大，中，小


if __name__ == "__main__":
    from conf import config

    yolov3 = YoloV3Net(config.DefaultCocoConfig)
    for key, value in yolov3.state_dict().items():
        print(key, value.shape)

    print(yolov3)

    yolov3.load_state_dict(torch.load("../weights/demo_yolov3_weights.pth"))

"""
backbone.init_conv.conv.weight torch.Size([32, 3, 3, 3])
backbone.init_conv.bn.weight torch.Size([32])
backbone.init_conv.bn.bias torch.Size([32])
backbone.init_conv.bn.running_mean torch.Size([32])
backbone.init_conv.bn.running_var torch.Size([32])
backbone.init_conv.bn.num_batches_tracked torch.Size([])
backbone.layer1.down_sample_conv.conv.weight torch.Size([64, 32, 3, 3])
backbone.layer1.down_sample_conv.bn.weight torch.Size([64])
backbone.layer1.down_sample_conv.bn.bias torch.Size([64])
backbone.layer1.down_sample_conv.bn.running_mean torch.Size([64])
backbone.layer1.down_sample_conv.bn.running_var torch.Size([64])
backbone.layer1.down_sample_conv.bn.num_batches_tracked torch.Size([])
backbone.layer1.residual_0.conv1.conv.weight torch.Size([32, 64, 1, 1])
backbone.layer1.residual_0.conv1.bn.weight torch.Size([32])
backbone.layer1.residual_0.conv1.bn.bias torch.Size([32])
backbone.layer1.residual_0.conv1.bn.running_mean torch.Size([32])
backbone.layer1.residual_0.conv1.bn.running_var torch.Size([32])
backbone.layer1.residual_0.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer1.residual_0.conv2.conv.weight torch.Size([64, 32, 3, 3])
backbone.layer1.residual_0.conv2.bn.weight torch.Size([64])
backbone.layer1.residual_0.conv2.bn.bias torch.Size([64])
backbone.layer1.residual_0.conv2.bn.running_mean torch.Size([64])
backbone.layer1.residual_0.conv2.bn.running_var torch.Size([64])
backbone.layer1.residual_0.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer2.down_sample_conv.conv.weight torch.Size([128, 64, 3, 3])
backbone.layer2.down_sample_conv.bn.weight torch.Size([128])
backbone.layer2.down_sample_conv.bn.bias torch.Size([128])
backbone.layer2.down_sample_conv.bn.running_mean torch.Size([128])
backbone.layer2.down_sample_conv.bn.running_var torch.Size([128])
backbone.layer2.down_sample_conv.bn.num_batches_tracked torch.Size([])
backbone.layer2.residual_0.conv1.conv.weight torch.Size([64, 128, 1, 1])
backbone.layer2.residual_0.conv1.bn.weight torch.Size([64])
backbone.layer2.residual_0.conv1.bn.bias torch.Size([64])
backbone.layer2.residual_0.conv1.bn.running_mean torch.Size([64])
backbone.layer2.residual_0.conv1.bn.running_var torch.Size([64])
backbone.layer2.residual_0.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer2.residual_0.conv2.conv.weight torch.Size([128, 64, 3, 3])
backbone.layer2.residual_0.conv2.bn.weight torch.Size([128])
backbone.layer2.residual_0.conv2.bn.bias torch.Size([128])
backbone.layer2.residual_0.conv2.bn.running_mean torch.Size([128])
backbone.layer2.residual_0.conv2.bn.running_var torch.Size([128])
backbone.layer2.residual_0.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer2.residual_1.conv1.conv.weight torch.Size([64, 128, 1, 1])
backbone.layer2.residual_1.conv1.bn.weight torch.Size([64])
backbone.layer2.residual_1.conv1.bn.bias torch.Size([64])
backbone.layer2.residual_1.conv1.bn.running_mean torch.Size([64])
backbone.layer2.residual_1.conv1.bn.running_var torch.Size([64])
backbone.layer2.residual_1.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer2.residual_1.conv2.conv.weight torch.Size([128, 64, 3, 3])
backbone.layer2.residual_1.conv2.bn.weight torch.Size([128])
backbone.layer2.residual_1.conv2.bn.bias torch.Size([128])
backbone.layer2.residual_1.conv2.bn.running_mean torch.Size([128])
backbone.layer2.residual_1.conv2.bn.running_var torch.Size([128])
backbone.layer2.residual_1.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.down_sample_conv.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.down_sample_conv.bn.weight torch.Size([256])
backbone.layer3.down_sample_conv.bn.bias torch.Size([256])
backbone.layer3.down_sample_conv.bn.running_mean torch.Size([256])
backbone.layer3.down_sample_conv.bn.running_var torch.Size([256])
backbone.layer3.down_sample_conv.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_0.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_0.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_0.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_0.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_0.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_0.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_0.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_0.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_0.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_0.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_0.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_0.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_1.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_1.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_1.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_1.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_1.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_1.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_1.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_1.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_1.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_1.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_1.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_1.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_2.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_2.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_2.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_2.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_2.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_2.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_2.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_2.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_2.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_2.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_2.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_2.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_3.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_3.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_3.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_3.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_3.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_3.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_3.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_3.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_3.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_3.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_3.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_3.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_4.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_4.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_4.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_4.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_4.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_4.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_4.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_4.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_4.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_4.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_4.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_4.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_5.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_5.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_5.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_5.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_5.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_5.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_5.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_5.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_5.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_5.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_5.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_5.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_6.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_6.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_6.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_6.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_6.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_6.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_6.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_6.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_6.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_6.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_6.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_6.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_7.conv1.conv.weight torch.Size([128, 256, 1, 1])
backbone.layer3.residual_7.conv1.bn.weight torch.Size([128])
backbone.layer3.residual_7.conv1.bn.bias torch.Size([128])
backbone.layer3.residual_7.conv1.bn.running_mean torch.Size([128])
backbone.layer3.residual_7.conv1.bn.running_var torch.Size([128])
backbone.layer3.residual_7.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer3.residual_7.conv2.conv.weight torch.Size([256, 128, 3, 3])
backbone.layer3.residual_7.conv2.bn.weight torch.Size([256])
backbone.layer3.residual_7.conv2.bn.bias torch.Size([256])
backbone.layer3.residual_7.conv2.bn.running_mean torch.Size([256])
backbone.layer3.residual_7.conv2.bn.running_var torch.Size([256])
backbone.layer3.residual_7.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.down_sample_conv.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.down_sample_conv.bn.weight torch.Size([512])
backbone.layer4.down_sample_conv.bn.bias torch.Size([512])
backbone.layer4.down_sample_conv.bn.running_mean torch.Size([512])
backbone.layer4.down_sample_conv.bn.running_var torch.Size([512])
backbone.layer4.down_sample_conv.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_0.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_0.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_0.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_0.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_0.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_0.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_0.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_0.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_0.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_0.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_0.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_0.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_1.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_1.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_1.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_1.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_1.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_1.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_1.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_1.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_1.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_1.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_1.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_1.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_2.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_2.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_2.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_2.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_2.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_2.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_2.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_2.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_2.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_2.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_2.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_2.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_3.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_3.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_3.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_3.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_3.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_3.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_3.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_3.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_3.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_3.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_3.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_3.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_4.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_4.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_4.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_4.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_4.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_4.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_4.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_4.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_4.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_4.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_4.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_4.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_5.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_5.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_5.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_5.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_5.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_5.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_5.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_5.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_5.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_5.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_5.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_5.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_6.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_6.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_6.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_6.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_6.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_6.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_6.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_6.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_6.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_6.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_6.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_6.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_7.conv1.conv.weight torch.Size([256, 512, 1, 1])
backbone.layer4.residual_7.conv1.bn.weight torch.Size([256])
backbone.layer4.residual_7.conv1.bn.bias torch.Size([256])
backbone.layer4.residual_7.conv1.bn.running_mean torch.Size([256])
backbone.layer4.residual_7.conv1.bn.running_var torch.Size([256])
backbone.layer4.residual_7.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer4.residual_7.conv2.conv.weight torch.Size([512, 256, 3, 3])
backbone.layer4.residual_7.conv2.bn.weight torch.Size([512])
backbone.layer4.residual_7.conv2.bn.bias torch.Size([512])
backbone.layer4.residual_7.conv2.bn.running_mean torch.Size([512])
backbone.layer4.residual_7.conv2.bn.running_var torch.Size([512])
backbone.layer4.residual_7.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer5.down_sample_conv.conv.weight torch.Size([1024, 512, 3, 3])
backbone.layer5.down_sample_conv.bn.weight torch.Size([1024])
backbone.layer5.down_sample_conv.bn.bias torch.Size([1024])
backbone.layer5.down_sample_conv.bn.running_mean torch.Size([1024])
backbone.layer5.down_sample_conv.bn.running_var torch.Size([1024])
backbone.layer5.down_sample_conv.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_0.conv1.conv.weight torch.Size([512, 1024, 1, 1])
backbone.layer5.residual_0.conv1.bn.weight torch.Size([512])
backbone.layer5.residual_0.conv1.bn.bias torch.Size([512])
backbone.layer5.residual_0.conv1.bn.running_mean torch.Size([512])
backbone.layer5.residual_0.conv1.bn.running_var torch.Size([512])
backbone.layer5.residual_0.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_0.conv2.conv.weight torch.Size([1024, 512, 3, 3])
backbone.layer5.residual_0.conv2.bn.weight torch.Size([1024])
backbone.layer5.residual_0.conv2.bn.bias torch.Size([1024])
backbone.layer5.residual_0.conv2.bn.running_mean torch.Size([1024])
backbone.layer5.residual_0.conv2.bn.running_var torch.Size([1024])
backbone.layer5.residual_0.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_1.conv1.conv.weight torch.Size([512, 1024, 1, 1])
backbone.layer5.residual_1.conv1.bn.weight torch.Size([512])
backbone.layer5.residual_1.conv1.bn.bias torch.Size([512])
backbone.layer5.residual_1.conv1.bn.running_mean torch.Size([512])
backbone.layer5.residual_1.conv1.bn.running_var torch.Size([512])
backbone.layer5.residual_1.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_1.conv2.conv.weight torch.Size([1024, 512, 3, 3])
backbone.layer5.residual_1.conv2.bn.weight torch.Size([1024])
backbone.layer5.residual_1.conv2.bn.bias torch.Size([1024])
backbone.layer5.residual_1.conv2.bn.running_mean torch.Size([1024])
backbone.layer5.residual_1.conv2.bn.running_var torch.Size([1024])
backbone.layer5.residual_1.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_2.conv1.conv.weight torch.Size([512, 1024, 1, 1])
backbone.layer5.residual_2.conv1.bn.weight torch.Size([512])
backbone.layer5.residual_2.conv1.bn.bias torch.Size([512])
backbone.layer5.residual_2.conv1.bn.running_mean torch.Size([512])
backbone.layer5.residual_2.conv1.bn.running_var torch.Size([512])
backbone.layer5.residual_2.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_2.conv2.conv.weight torch.Size([1024, 512, 3, 3])
backbone.layer5.residual_2.conv2.bn.weight torch.Size([1024])
backbone.layer5.residual_2.conv2.bn.bias torch.Size([1024])
backbone.layer5.residual_2.conv2.bn.running_mean torch.Size([1024])
backbone.layer5.residual_2.conv2.bn.running_var torch.Size([1024])
backbone.layer5.residual_2.conv2.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_3.conv1.conv.weight torch.Size([512, 1024, 1, 1])
backbone.layer5.residual_3.conv1.bn.weight torch.Size([512])
backbone.layer5.residual_3.conv1.bn.bias torch.Size([512])
backbone.layer5.residual_3.conv1.bn.running_mean torch.Size([512])
backbone.layer5.residual_3.conv1.bn.running_var torch.Size([512])
backbone.layer5.residual_3.conv1.bn.num_batches_tracked torch.Size([])
backbone.layer5.residual_3.conv2.conv.weight torch.Size([1024, 512, 3, 3])
backbone.layer5.residual_3.conv2.bn.weight torch.Size([1024])
backbone.layer5.residual_3.conv2.bn.bias torch.Size([1024])
backbone.layer5.residual_3.conv2.bn.running_mean torch.Size([1024])
backbone.layer5.residual_3.conv2.bn.running_var torch.Size([1024])
backbone.layer5.residual_3.conv2.bn.num_batches_tracked torch.Size([])
last_layer_13.0.conv.weight torch.Size([512, 1024, 1, 1])
last_layer_13.0.bn.weight torch.Size([512])
last_layer_13.0.bn.bias torch.Size([512])
last_layer_13.0.bn.running_mean torch.Size([512])
last_layer_13.0.bn.running_var torch.Size([512])
last_layer_13.0.bn.num_batches_tracked torch.Size([])
last_layer_13.1.conv.weight torch.Size([1024, 512, 3, 3])
last_layer_13.1.bn.weight torch.Size([1024])
last_layer_13.1.bn.bias torch.Size([1024])
last_layer_13.1.bn.running_mean torch.Size([1024])
last_layer_13.1.bn.running_var torch.Size([1024])
last_layer_13.1.bn.num_batches_tracked torch.Size([])
last_layer_13.2.conv.weight torch.Size([512, 1024, 1, 1])
last_layer_13.2.bn.weight torch.Size([512])
last_layer_13.2.bn.bias torch.Size([512])
last_layer_13.2.bn.running_mean torch.Size([512])
last_layer_13.2.bn.running_var torch.Size([512])
last_layer_13.2.bn.num_batches_tracked torch.Size([])
last_layer_13.3.conv.weight torch.Size([1024, 512, 3, 3])
last_layer_13.3.bn.weight torch.Size([1024])
last_layer_13.3.bn.bias torch.Size([1024])
last_layer_13.3.bn.running_mean torch.Size([1024])
last_layer_13.3.bn.running_var torch.Size([1024])
last_layer_13.3.bn.num_batches_tracked torch.Size([])
last_layer_13.4.conv.weight torch.Size([512, 1024, 1, 1])
last_layer_13.4.bn.weight torch.Size([512])
last_layer_13.4.bn.bias torch.Size([512])
last_layer_13.4.bn.running_mean torch.Size([512])
last_layer_13.4.bn.running_var torch.Size([512])
last_layer_13.4.bn.num_batches_tracked torch.Size([])
last_layer_13.5.conv.weight torch.Size([1024, 512, 3, 3])
last_layer_13.5.bn.weight torch.Size([1024])
last_layer_13.5.bn.bias torch.Size([1024])
last_layer_13.5.bn.running_mean torch.Size([1024])
last_layer_13.5.bn.running_var torch.Size([1024])
last_layer_13.5.bn.num_batches_tracked torch.Size([])
last_layer_13.6.weight torch.Size([255, 1024, 1, 1])
last_layer_13.6.bias torch.Size([255])
last_layer_26_conv.conv.weight torch.Size([256, 512, 1, 1])
last_layer_26_conv.bn.weight torch.Size([256])
last_layer_26_conv.bn.bias torch.Size([256])
last_layer_26_conv.bn.running_mean torch.Size([256])
last_layer_26_conv.bn.running_var torch.Size([256])
last_layer_26_conv.bn.num_batches_tracked torch.Size([])
last_layer_26.0.conv.weight torch.Size([256, 768, 1, 1])
last_layer_26.0.bn.weight torch.Size([256])
last_layer_26.0.bn.bias torch.Size([256])
last_layer_26.0.bn.running_mean torch.Size([256])
last_layer_26.0.bn.running_var torch.Size([256])
last_layer_26.0.bn.num_batches_tracked torch.Size([])
last_layer_26.1.conv.weight torch.Size([512, 256, 3, 3])
last_layer_26.1.bn.weight torch.Size([512])
last_layer_26.1.bn.bias torch.Size([512])
last_layer_26.1.bn.running_mean torch.Size([512])
last_layer_26.1.bn.running_var torch.Size([512])
last_layer_26.1.bn.num_batches_tracked torch.Size([])
last_layer_26.2.conv.weight torch.Size([256, 512, 1, 1])
last_layer_26.2.bn.weight torch.Size([256])
last_layer_26.2.bn.bias torch.Size([256])
last_layer_26.2.bn.running_mean torch.Size([256])
last_layer_26.2.bn.running_var torch.Size([256])
last_layer_26.2.bn.num_batches_tracked torch.Size([])
last_layer_26.3.conv.weight torch.Size([512, 256, 3, 3])
last_layer_26.3.bn.weight torch.Size([512])
last_layer_26.3.bn.bias torch.Size([512])
last_layer_26.3.bn.running_mean torch.Size([512])
last_layer_26.3.bn.running_var torch.Size([512])
last_layer_26.3.bn.num_batches_tracked torch.Size([])
last_layer_26.4.conv.weight torch.Size([256, 512, 1, 1])
last_layer_26.4.bn.weight torch.Size([256])
last_layer_26.4.bn.bias torch.Size([256])
last_layer_26.4.bn.running_mean torch.Size([256])
last_layer_26.4.bn.running_var torch.Size([256])
last_layer_26.4.bn.num_batches_tracked torch.Size([])
last_layer_26.5.conv.weight torch.Size([512, 256, 3, 3])
last_layer_26.5.bn.weight torch.Size([512])
last_layer_26.5.bn.bias torch.Size([512])
last_layer_26.5.bn.running_mean torch.Size([512])
last_layer_26.5.bn.running_var torch.Size([512])
last_layer_26.5.bn.num_batches_tracked torch.Size([])
last_layer_26.6.weight torch.Size([255, 512, 1, 1])
last_layer_26.6.bias torch.Size([255])
last_layer_52_conv.conv.weight torch.Size([128, 256, 1, 1])
last_layer_52_conv.bn.weight torch.Size([128])
last_layer_52_conv.bn.bias torch.Size([128])
last_layer_52_conv.bn.running_mean torch.Size([128])
last_layer_52_conv.bn.running_var torch.Size([128])
last_layer_52_conv.bn.num_batches_tracked torch.Size([])
last_layer_52.0.conv.weight torch.Size([128, 384, 1, 1])
last_layer_52.0.bn.weight torch.Size([128])
last_layer_52.0.bn.bias torch.Size([128])
last_layer_52.0.bn.running_mean torch.Size([128])
last_layer_52.0.bn.running_var torch.Size([128])
last_layer_52.0.bn.num_batches_tracked torch.Size([])
last_layer_52.1.conv.weight torch.Size([256, 128, 3, 3])
last_layer_52.1.bn.weight torch.Size([256])
last_layer_52.1.bn.bias torch.Size([256])
last_layer_52.1.bn.running_mean torch.Size([256])
last_layer_52.1.bn.running_var torch.Size([256])
last_layer_52.1.bn.num_batches_tracked torch.Size([])
last_layer_52.2.conv.weight torch.Size([128, 256, 1, 1])
last_layer_52.2.bn.weight torch.Size([128])
last_layer_52.2.bn.bias torch.Size([128])
last_layer_52.2.bn.running_mean torch.Size([128])
last_layer_52.2.bn.running_var torch.Size([128])
last_layer_52.2.bn.num_batches_tracked torch.Size([])
last_layer_52.3.conv.weight torch.Size([256, 128, 3, 3])
last_layer_52.3.bn.weight torch.Size([256])
last_layer_52.3.bn.bias torch.Size([256])
last_layer_52.3.bn.running_mean torch.Size([256])
last_layer_52.3.bn.running_var torch.Size([256])
last_layer_52.3.bn.num_batches_tracked torch.Size([])
last_layer_52.4.conv.weight torch.Size([128, 256, 1, 1])
last_layer_52.4.bn.weight torch.Size([128])
last_layer_52.4.bn.bias torch.Size([128])
last_layer_52.4.bn.running_mean torch.Size([128])
last_layer_52.4.bn.running_var torch.Size([128])
last_layer_52.4.bn.num_batches_tracked torch.Size([])
last_layer_52.5.conv.weight torch.Size([256, 128, 3, 3])
last_layer_52.5.bn.weight torch.Size([256])
last_layer_52.5.bn.bias torch.Size([256])
last_layer_52.5.bn.running_mean torch.Size([256])
last_layer_52.5.bn.running_var torch.Size([256])
last_layer_52.5.bn.num_batches_tracked torch.Size([])
last_layer_52.6.weight torch.Size([255, 256, 1, 1])
last_layer_52.6.bias torch.Size([255])
YoloV3Net(
  (backbone): DarkNet(
    (init_conv): Conv2d(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (layer1): Sequential(
      (down_sample_conv): Conv2d(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): LeakyReLU(negative_slope=0.1)
      )
      (residual_0): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
    )
    (layer2): Sequential(
      (down_sample_conv): Conv2d(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): LeakyReLU(negative_slope=0.1)
      )
      (residual_0): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_1): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
    )
    (layer3): Sequential(
      (down_sample_conv): Conv2d(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): LeakyReLU(negative_slope=0.1)
      )
      (residual_0): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_1): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_2): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_3): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_4): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_5): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_6): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_7): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
    )
    (layer4): Sequential(
      (down_sample_conv): Conv2d(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): LeakyReLU(negative_slope=0.1)
      )
      (residual_0): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_1): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_2): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_3): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_4): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_5): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_6): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_7): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
    )
    (layer5): Sequential(
      (down_sample_conv): Conv2d(
        (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): LeakyReLU(negative_slope=0.1)
      )
      (residual_0): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_1): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_2): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
      (residual_3): BasicBlock(
        (conv1): Conv2d(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
        (conv2): Conv2d(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): LeakyReLU(negative_slope=0.1)
        )
      )
    )
  )
  (last_layer_13): ModuleList(
    (0): Conv2d(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (1): Conv2d(
      (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (2): Conv2d(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (3): Conv2d(
      (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (4): Conv2d(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (5): Conv2d(
      (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (6): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
  )
  (last_layer_26_conv): Conv2d(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.1)
  )
  (last_layer_26_upsample): Upsample(scale_factor=2.0, mode=nearest)
  (last_layer_26): ModuleList(
    (0): Conv2d(
      (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (1): Conv2d(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (2): Conv2d(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (3): Conv2d(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (4): Conv2d(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (5): Conv2d(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (6): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
  )
  (last_layer_52_conv): Conv2d(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.1)
  )
  (last_layer_52_upsample): Upsample(scale_factor=2.0, mode=nearest)
  (last_layer_52): ModuleList(
    (0): Conv2d(
      (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (1): Conv2d(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (2): Conv2d(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (3): Conv2d(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (4): Conv2d(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (5): Conv2d(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): LeakyReLU(negative_slope=0.1)
    )
    (6): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
  )
)
"""