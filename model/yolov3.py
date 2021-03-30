import os

import numpy
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
import torch.nn as nn

from model import yolov3net
from util import yolo_utils

import model.yolov3decode

import dataset.dataset_utils


# -----------------------------------------------------------------------------------------------------------#
# class YoloV3(object) # YoloV3 预测网络
# -----------------------------------------------------------------------------------------------------------#

class YoloV3(object):
    """
    对 YoloV3 的三层预测结果提供解析
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config
        self.generate()

    def generate(self):
        """
        初始化预测模型和解析工具
        :return:
        """
        print("YoloV3 generate...")
        # 1. 生成模型
        self.net = yolov3net.YoloV3Net(self.config)
        # 2. 加载模型权重
        device = torch.device('cuda') if self.config["cuda"] else torch.device('cpu')
        state_dict = torch.load(self.config["weights_path"], map_location=device)
        self.net.load_state_dict(state_dict)
        # 3. 网络开启 eval 模式
        self.net = self.net.eval()
        # 4. 初始化特征层解码器
        self.yolov3_decode = model.yolov3decode.YoloV3Decode(config=self.config)
        print("YoloV3 generate Success")

    def predict(self, tensord_image: torch.Tensor, tensord_target: torch.Tensor = None) -> Image.Image:
        """
        检测图片

        :param image: 待检测图片，单张
        :return: 检测后图片，绘制标签和预测框
        """
        with torch.no_grad():  # 没有梯度传递，进行图像检测
            # 将图像输入网络当中进行预测
            outputs = self.net(tensord_image.unsqueeze(dim=0))
            # 预测框列表
            output_list = []
            # 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                output_list.append(self.yolov3_decode(outputs[index]))

            # 将预测框进行堆叠
            output = torch.cat(output_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            # 进行非极大抑制
            # (batch, 13*13*3 + 26*26*3 + 52*52*3 = 10647, (4+1+classes)) ->
            # (box_num, x1 + y1 + x2 + y2 + obj_conf + class_conf + class_label = 7)
            batch_detections = model.yolov3decode.non_max_suppression(
                output,  # 预测框列表
                conf_threshold=self.config["conf_threshold"],  # 置信度
                nms_iou_threshold=self.config["nms_iou_threshold"]  # iou 阈值
            )

            # 检测图片时一个批次就一个图片，因此取出第 0 个
            batch_detections = batch_detections[0].cpu().numpy()

            top_boxes = numpy.array(batch_detections[:, :4])  # 预测框

        image = Image.fromarray(
            (numpy.transpose(tensord_image.numpy(), (1, 2, 0)) * 255).astype(numpy.uint8),
            mode="RGB"
        )

        # -----------------------------------------------------------------------------------------------------------#
        # 绘制预测框
        # -----------------------------------------------------------------------------------------------------------#
        print("predict_boxes:", top_boxes)

        # 对每一个类别分别绘制预测框（红）
        for box in numpy.around(numpy.asarray(batch_detections)).astype(numpy.int):
            (xmin, ymin, xmax, ymax, conf, label) = box
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000")
            # 绘制标签
            font = ImageFont.truetype('/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf', 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
            del draw

        # -----------------------------------------------------------------------------------------------------------#
        # 绘制真值框
        # -----------------------------------------------------------------------------------------------------------#

        if tensord_target != None:
            # 解析真值框
            truth_boxes = dataset.dataset_utils.decode_tensord_target(self.config, tensord_target)
            print("truth_boxes:", truth_boxes)

            # 绘制真值框（绿）
            for truth_box in truth_boxes:
                (xmin, ymin, xmax, ymax, label) = truth_box
                draw = ImageDraw.Draw(image)
                draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00")
                # 绘制标签
                font = ImageFont.truetype('/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf', 32)
                draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#00FF00")
                del draw

        return image
