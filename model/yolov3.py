import os

import numpy
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
from typing import List

import model.yolov3net
import model.yolov3decode
import dataset.transform


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
        self.cuda = self.config["cuda"]
        self.generate()

    def generate(self):
        """
        初始化预测模型和解析工具
        :return:
        """
        print("YoloV3 generate...")
        # 1. 生成模型
        self.net = model.yolov3net.YoloV3Net(self.config)
        if self.cuda:
            self.net = self.net.cuda()
        # 2. 加载模型权重
        device = torch.device("cuda") if self.config["cuda"] else torch.device("cpu")
        state_dict = torch.load(self.config["weights_path"], map_location=device)
        self.net.load_state_dict(state_dict)
        # 3. 网络开启 eval 模式
        self.net = self.net.eval()
        # 4. 初始化特征层解码器
        self.yolov3_decode = model.yolov3decode.YoloV3Decode(config=self.config)
        # 5. 预测结果恢复变换
        self.rescale_boxes = dataset.transform.RescaleBoxes(config=self.config)
        print("YoloV3 generate Success")

    def predict(self, tensord_image: torch.Tensor, tensord_target: torch.Tensor = None) -> Image.Image:
        """
        检测图片

        :param image: 待检测图片，单张
        :return: 检测后图片，绘制标签和预测框
        """
        # -----------------------------------------------------------------------------------------------------------#
        # step1. 提取预测框
        # -----------------------------------------------------------------------------------------------------------#
        with torch.no_grad():  # 1. 没有梯度传递，进行图像检测
            # 2. 将图像输入网络当中进行预测
            predict_feature_list = self.net(tensord_image.unsqueeze(dim=0))
            predict_bbox_attrs_list = []  # 预测框列表

            # 3. 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                predict_bbox_attrs_list.append(self.yolov3_decode(predict_feature_list[index]))

            # 4. 将预测框进行堆叠
            predict_bbox_attrs = torch.cat(predict_bbox_attrs_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            # 5. 进行非极大抑制
            predict_bbox_attrs_after_nms = model.yolov3decode.non_max_suppression(
                predict_bbox_attrs,  # 预测框列表
                conf_threshold=self.config["conf_threshold"],  # 置信度
                nms_iou_threshold=self.config["nms_iou_threshold"]  # iou 阈值
            )

        # -----------------------------------------------------------------------------------------------------------#
        # step2. 转为 PIL.Image.Image
        # -----------------------------------------------------------------------------------------------------------#
        image = torchvision.transforms.ToPILImage()(tensord_image)
        # image = Image.fromarray(
        #     (
        #             numpy.transpose(
        #                 tensord_image.numpy(),
        #                 (1, 2, 0)
        #             ) * 255
        #     ).astype(numpy.uint8),
        #     mode="RGB"
        # )

        # -----------------------------------------------------------------------------------------------------------#
        # step3. 绘制预测框
        # -----------------------------------------------------------------------------------------------------------#
        print("predict_boxes:\n", predict_bbox_attrs_after_nms)

        if predict_bbox_attrs_after_nms[0] == None:
            return image

        # 1. 绘制预测框（红）
        for predict_box in numpy.around(predict_bbox_attrs_after_nms[0].numpy()).astype(numpy.int):
            (xmin, ymin, xmax, ymax, conf, label) = predict_box
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000")
            # 绘制标签
            font = ImageFont.truetype("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf", 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
            del draw

        # -----------------------------------------------------------------------------------------------------------#
        # step4. 绘制真值框
        # -----------------------------------------------------------------------------------------------------------#
        if tensord_target != None:
            # 1. 解析真值框
            truth_boxes = model.yolov3decode.decode_tensord_target(self.config, tensord_target)
            print("truth_boxes:\n", truth_boxes)

            # 2. 绘制真值框（绿）
            for truth_box in truth_boxes:
                (xmin, ymin, xmax, ymax, label) = truth_box
                draw = ImageDraw.Draw(image)
                draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00")
                # 绘制标签
                font = ImageFont.truetype("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf", 32)
                draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#00FF00")
                del draw

        return image

    def predict_annotation(self, tensord_image: torch.Tensor) -> List[tuple]:
        """
        检测图片

        :param image: 待检测图片，单张
        """
        # -----------------------------------------------------------------------------------------------------------#
        # step1. 提取预测框
        # -----------------------------------------------------------------------------------------------------------#
        with torch.no_grad():  # 1. 没有梯度传递，进行图像检测
            # 2. 将图像输入网络当中进行预测
            predict_feature_list = self.net(tensord_image.unsqueeze(dim=0))
            predict_bbox_attrs_list = []  # 预测框列表

            # 3. 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                predict_bbox_attrs_list.append(self.yolov3_decode(predict_feature_list[index]))

            # 4. 将预测框进行堆叠
            predict_bbox_attrs = torch.cat(predict_bbox_attrs_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            # 5. 进行非极大抑制
            predict_bbox_attrs_after_nms = model.yolov3decode.non_max_suppression(
                predict_bbox_attrs,  # 预测框列表
                conf_threshold=self.config["conf_threshold"],  # 置信度
                nms_iou_threshold=self.config["nms_iou_threshold"]  # iou 阈值
            )

        # -----------------------------------------------------------------------------------------------------------#
        # step3. 绘制预测框
        # -----------------------------------------------------------------------------------------------------------#
        predict_annotation = []
        for predict_box in predict_bbox_attrs_after_nms[0]:
            (xmin, ymin, xmax, ymax, conf, label) = predict_box
            predict_annotation.append((conf, xmin, ymin, xmax, ymax, label))

        return predict_annotation

    def predict_with_truth_annotation(self, tensord_image: torch.Tensor, truth_annotation: dict) -> Image.Image:
        # -----------------------------------------------------------------------------------------------------------#
        # step1. 提取预测框
        # -----------------------------------------------------------------------------------------------------------#
        with torch.no_grad():  # 1. 没有梯度传递，进行图像检测
            # 2. 将图像输入网络当中进行预测
            predict_feature_list = self.net(tensord_image.unsqueeze(dim=0))
            predict_bbox_attrs_list = []  # 预测框列表

            # 3. 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                predict_bbox_attrs_list.append(self.yolov3_decode(predict_feature_list[index]))

            # 4. 将预测框进行堆叠
            predict_bbox_attrs = torch.cat(predict_bbox_attrs_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            # 5. 进行非极大抑制
            predict_bbox_attrs_after_nms = model.yolov3decode.non_max_suppression(
                predict_bbox_attrs,  # 预测框列表
                conf_threshold=self.config["conf_threshold"],  # 置信度
                nms_iou_threshold=self.config["nms_iou_threshold"]  # iou 阈值
            )

        # -----------------------------------------------------------------------------------------------------------#
        # step2. 转为 PIL.Image.Image
        # -----------------------------------------------------------------------------------------------------------#
        image = truth_annotation["raw_image"]

        # -----------------------------------------------------------------------------------------------------------#
        # step3. 绘制预测框
        # -----------------------------------------------------------------------------------------------------------#
        detection_result = open(
            os.path.join(
                "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/map_input/detection_results",
                truth_annotation["filename"].split(".")[0] + ".txt"
            ),
            "w"
        )

        if predict_bbox_attrs_after_nms[0] == None:
            return image

        # 1. 解析预测框
        image, predict_boxes = self.rescale_boxes(image, predict_bbox_attrs_after_nms[0].numpy())
        print("predict_boxes:\n", predict_boxes)

        # 1. 绘制预测框（红）
        for predict_box in predict_boxes:
            (xmin, ymin, xmax, ymax, conf, label) = predict_box
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000")
            # 绘制标签
            font = ImageFont.truetype("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf", 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
            del draw

            detection_result.write("%s %s %s %s %s %s\n" % (self.config["labels"][label], conf, xmin, ymin, xmax, ymax))

        # -----------------------------------------------------------------------------------------------------------#
        # step4. 绘制真值框
        # -----------------------------------------------------------------------------------------------------------#
        # 1. 解析真值框
        print("truth_boxes:\n", truth_annotation["boxes"])

        # 2. 绘制真值框（绿）
        for truth_box in truth_annotation["boxes"]:
            (xmin, ymin, xmax, ymax, label) = truth_box
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00")
            # 绘制标签
            font = ImageFont.truetype("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf", 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#00FF00")
            del draw

        return image

    def predict_detection_results(self, tensord_image: torch.Tensor, truth_annotation: dict) -> None:
        # -----------------------------------------------------------------------------------------------------------#
        # step1. 提取预测框
        # -----------------------------------------------------------------------------------------------------------#
        # with torch.no_grad():  # 1. 没有梯度传递，进行图像检测
        #     # 2. 将图像输入网络当中进行预测
        #     tensord_images = tensord_image.unsqueeze(dim=0)
        #     if self.cuda:
        #         tensord_images = tensord_images.cuda()
        #     predict_feature_list = self.net(tensord_images)
        #     predict_bbox_attrs_list = []  # 预测框列表
        #
        #     # 3. 对三种大小的预测框进行解析
        #     for index in range(3):  # 顺序：大，中，小
        #         predict_bbox_attrs_list.append(self.yolov3_decode(predict_feature_list[index]))
        #
        #     # 4. 将预测框进行堆叠
        #     predict_bbox_attrs = torch.cat(predict_bbox_attrs_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647
        #
        #     # 5. 进行非极大抑制
        #     predict_bbox_attrs_after_nms = model.yolov3decode.non_max_suppression(
        #         predict_bbox_attrs,  # 预测框列表
        #         conf_threshold=self.config["conf_threshold"],  # 置信度
        #         nms_iou_threshold=self.config["nms_iou_threshold"]  # iou 阈值
        #     )

        # -----------------------------------------------------------------------------------------------------------#
        # step2. 转为 PIL.Image.Image
        # -----------------------------------------------------------------------------------------------------------#
        image = truth_annotation["raw_image"]

        # -----------------------------------------------------------------------------------------------------------#
        # step3. 记录预测框和真值框
        # -----------------------------------------------------------------------------------------------------------#
        # detection_result = open(
        #     os.path.join(
        #         os.getcwd(),
        #         "map_input",
        #         "detection_result",
        #         truth_annotation["filename"].split(".")[0] + ".txt"
        #     ),
        #     "w"
        # )

        ground_truth = open(
            os.path.join(
                os.getcwd(),
                "map_input",
                "ground_truth",
                truth_annotation["filename"].split(".")[0] + ".txt"
            ),
            "w"
        )

        # if predict_bbox_attrs_after_nms[0] == None:
        #     return image
        #
        # # 1. 解析预测框
        # image, predict_boxes = self.rescale_boxes(image, predict_bbox_attrs_after_nms[0].cpu().numpy())
        #
        # # 1. 绘制预测框（红）
        # for predict_box in predict_boxes:
        #     (xmin, ymin, xmax, ymax, conf, label) = predict_box
        #     detection_result.write("%s %s %s %s %s %s\n" % (self.config["labels"][label], conf, xmin, ymin, xmax, ymax))

        # 2. 绘制真值框（绿）
        for truth_box in truth_annotation["boxes"]:
            (xmin, ymin, xmax, ymax, label) = truth_box
            ground_truth.write("%s %s %s %s %s\n" % (self.config["labels"][label], xmin, ymin, xmax, ymax))
