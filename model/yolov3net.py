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
