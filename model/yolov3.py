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
        self.cuda = self.config["cuda"] and torch.cuda.is_available()
        self.generate()

    def generate(self):
        """
        初始化预测模型和解析工具
        :return:
        """
        print("YoloV3 generate...")
        # 生成模型
        self.net = yolov3net.YoloV3Net(self.config)
        # 加载模型权重
        device = torch.device('cuda') if self.cuda else torch.device('cpu')
        state_dict = torch.load(self.config["weights_path"], map_location=device)
        self.net.load_state_dict(state_dict)
        # 网络开启 eval 模式
        self.net = self.net.eval()
        # 网络开启 GPU 模式
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        # 初始化三个特征层解码用的工具
        self.yolov3_decodes = []  # 大，中，小
        for i in range(3):
            self.yolov3_decodes.append(
                yolo_utils.DecodeBox(
                    self.config["anchors"][i],
                    self.config["classes"],
                    self.config["image_height"],  # height
                    self.config["image_width"],  # width
                    self.cuda
                )
            )

        self.yolov3_decode = model.yolov3decode.YoloV3Decode(config=self.config)
        print("YoloV3 generate Success")

    def predict(self, tensord_image: torch.Tensor, tensord_target: torch.Tensor) -> Image.Image:
        """
        检测图片
        :param image: 待检测图片，单张
        :return: 检测后图片，绘制标签和预测框
        """
        with torch.no_grad():  # 没有梯度传递，进行图像检测
            # 将图像输入网络当中进行预测
            outputs = self.net(torch.unsqueeze(tensord_image, dim=0))
            # 预测框列表
            output_list = []
            # 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                # output_list.append(self.yolov3_decodes[index](outputs[index]))
                output_list.append(self.yolov3_decode(outputs[index]))

            """
            for i, o in enumerate(output_list): print("output_list", i, ":", o.shape)
            output_list 0 : torch.Size([1, 507, 85])，13*13*3 = 507
            output_list 1 : torch.Size([1, 2028, 85])，26*26*3=2028
            output_list 2 : torch.Size([1, 8112, 85])，52*52*3=8112
            """

            # 将预测框进行堆叠
            output = torch.cat(output_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            """
            print("output:", output.shape)  # torch.Size([1, 10647, 85])
            """

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

            # box_num, x1 + y1 + x2 + y2 + obj_conf + class_conf + class_pred
            # print("batch_detections:", batch_detections.shape)  # (12, 7)

            # 对预测框进行得分筛选
            top_label = numpy.array(batch_detections[:, -1], numpy.int32)  # 标签
            top_boxes = numpy.array(batch_detections[:, :4])  # 预测框

        image = Image.fromarray(
            (numpy.transpose(tensord_image.numpy(), (1, 2, 0)) * 255).astype(numpy.uint8),
            mode="RGB"
        )

        # -----------------------------------------------------------------------------------------------------------#
        # 绘制真值框
        # -----------------------------------------------------------------------------------------------------------#

        # 解析真值框
        raw_boxes = dataset.dataset_utils.decode_tensord_target(self.config, tensord_target)
        print("raw_boxes:", raw_boxes)

        # 绘制真值框（绿）
        for raw_box in raw_boxes:
            (xmin, ymin, xmax, ymax, label) = raw_box
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00")
            # 绘制标签
            font = ImageFont.truetype('/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf', 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#00FF00")
            del draw

        # -----------------------------------------------------------------------------------------------------------#
        # 绘制预测框
        # -----------------------------------------------------------------------------------------------------------#
        print("predict_boxes:", top_boxes)

        # 对每一个类别分别绘制预测框（红）
        for box in numpy.around(numpy.asarray(batch_detections)).astype(numpy.int):
            (xmin, ymin, xmax, ymax, _, label) = box
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000")
            # 绘制标签
            font = ImageFont.truetype('/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/assets/simhei.ttf', 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
            del draw
        # for index, label in enumerate(top_label):
        # xmin, ymin, xmax, ymax = top_boxes[index]
        #
        # # 绘制一个稍大一点的框框
        # ymin = ymin - 5
        # xmin = xmin - 5
        # ymax = ymax + 5
        # xmax = xmax + 5
        # # 过滤框大小
        # ymin = max(0, numpy.floor(ymin + 0.5).astype('int32'))
        # xmin = max(0, numpy.floor(xmin + 0.5).astype('int32'))
        # ymax = min(self.config["image_height"], numpy.floor(ymax + 0.5).astype('int32'))
        # xmax = min(self.config["image_width"], numpy.floor(xmax + 0.5).astype('int32'))
        #
        # print("draw: box:", xmin, ymin, xmax, ymax, "label:", self.config["labels"][label])
        #
        # # 画框框
        # draw = ImageDraw.Draw(image)
        # draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000")
        # # 绘制标签
        # font = ImageFont.truetype('/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/simhei.ttf', 32)
        # draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
        # del draw

        return image

    def detect_image(self, image: Image.Image) -> Image.Image:
        """
        检测图片
        :param image: 待检测图片，单张
        :return: 检测后图片，绘制标签和预测框
        """
        image_shape = numpy.array(numpy.shape(image)[0:2])  # 保存原始的图像大小 width * height
        image_raw_width = image_shape[0]
        image_raw_height = image_shape[1]

        # 给图像增加灰条，实现不失真的resize
        crop_image, _ = yolo_utils.letterbox_image(
            image,
            # 1000,
            self.config["image_width"],
            self.config["image_height"]
        )
        # width * height * RGB
        crop_image.show()
        # print(crop_image.size)
        # print(crop_image.getpixel((500, 100)))
        crop_image = numpy.array(crop_image)  # -> height * width * RGB
        # print(crop_image[100][500])

        # display_utils.show_numpy_image(crop_image, [0])
        # display_utils.show_numpy_image(crop_image, [1])
        # display_utils.show_numpy_image(crop_image, [2])

        # print(crop_image.shape)

        photo = torchvision.transforms.ToTensor()(crop_image)

        # 输入图像归一化到 0~1
        # photo = numpy.array(crop_image, dtype=numpy.float32) / 255.0

        # print(photo.shape)
        # photo = numpy.transpose(photo, (2, 0, 1))  # -> channels(RGB) * height * width
        # print(photo.shape)
        # print(photo[0][100][500])
        # print(photo[1][100][500])
        # print(photo[2][100][500])

        # 添加上batch_size维度
        images = [numpy.asarray(photo)]

        with torch.no_grad():  # 没有梯度传递，进行图像检测
            # print(photo.shape)
            images = torch.as_tensor(images)  # 输入图片转化为 torch Tensor
            print(images.shape)
            # print(images[0][0][100][500])
            # print(images[0][1][100][500])
            # print(images[0][2][100][500])
            # exit(-1)
            if self.cuda:  # 配置 GPU 且 Cuda 可用
                images = images.cuda()

            # 将图像输入网络当中进行预测
            outputs = self.net(images)
            # 预测框列表
            output_list = []
            # 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                output_list.append(self.yolov3_decodes[index](outputs[index]))

            """
            for i, o in enumerate(output_list): print("output_list", i, ":", o.shape)
            output_list 0 : torch.Size([1, 507, 85])，13*13*3 = 507
            output_list 1 : torch.Size([1, 2028, 85])，26*26*3=2028
            output_list 2 : torch.Size([1, 8112, 85])，52*52*3=8112
            """

            # 将预测框进行堆叠
            output = torch.cat(output_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            """
            print("output:", output.shape)  # torch.Size([1, 10647, 85])
            """

            # 进行非极大抑制
            # (batch, 13*13*3 + 26*26*3 + 52*52*3 = 10647, (4+1+classes)) ->
            # (box_num, x1 + y1 + x2 + y2 + obj_conf + class_conf + class_label = 7)
            batch_detections = yolo_utils.non_max_suppression(
                output,  # 预测框列表
                self.config["classes"],  # 类别数目
                conf_threshold=self.config["confidence"],  # 置信度
                nms_iou_threshold=self.config["iou"]  # iou 阈值
            )

            # 检测图片时一个批次就一个图片，因此取出第 0 个
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # box_num, x1 + y1 + x2 + y2 + obj_conf + class_conf + class_pred
            # print("batch_detections:", batch_detections.shape)  # (12, 7)

            # 对预测框进行得分筛选
            top_label = numpy.array(batch_detections[:, -1], numpy.int32)  # 标签
            top_boxes = numpy.array(batch_detections[:, :4])  # 预测框

            # (box_num) -> (box_num, 1) 增加一个维度
            top_xmin, top_ymin, top_xmax, top_ymax = numpy.expand_dims(top_boxes[:, 0], -1), \
                                                     numpy.expand_dims(top_boxes[:, 1], -1), \
                                                     numpy.expand_dims(top_boxes[:, 2], -1), \
                                                     numpy.expand_dims(top_boxes[:, 3], -1)

            # 从灰条检测图像中恢复原始的检测框
            boxes = yolo_utils.yolo_correct_boxes(
                top_xmin, top_ymin, top_xmax, top_ymax,
                self.config["image_width"], self.config["image_height"],
                image_raw_width, image_raw_height)

        # 对每一个类别分别绘制预测框
        for index, label in enumerate(top_label):
            ymin, xmin, ymax, xmax = boxes[index]

            print("raw: box:", xmin, ymin, xmax, ymax, "label:", self.config["labels"][label])

            # 绘制一个稍大一点的框框
            ymin = ymin - 5
            xmin = xmin - 5
            ymax = ymax + 5
            xmax = xmax + 5
            # 过滤框大小
            ymin = max(0, numpy.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, numpy.floor(xmin + 0.5).astype('int32'))
            ymax = min(image_raw_height, numpy.floor(ymax + 0.5).astype('int32'))
            xmax = min(image_raw_width, numpy.floor(xmax + 0.5).astype('int32'))

            print("draw: box:", xmin, ymin, xmax, ymax, "label:", self.config["labels"][label])

            # 画框框
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax])
            # 绘制标签
            font = ImageFont.truetype('/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/simhei.ttf', 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
            del draw

        return image


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    from conf import config

    config.PennFudanConfig["weights_path"] \
        = "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/logs/" \
          "Epoch100-Total_Loss0.3207-Val_Loss0.3198.pth"

    config.PennFudanConfig["confidence"] = 0.3

    yolov3 = YoloV3(config.PennFudanConfig)

    print("../images/test0.png")  # height: 415, width: 453
    print("../images/test_r.png")
    print("../images/test_g.png")
    print("../images/test_b.png")
    print("../images/street.jpg")
    print("../images/test_360.jpg")
    # print("../images/test_360_b.png")
    # print("../images/test_360_g.png")
    # print("../images/test_360_r.png")

    while True:
        image_path = input('Input image filename:')
        try:
            raw_image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            predict_image = yolov3.detect_image(raw_image)
            predict_image.show()
