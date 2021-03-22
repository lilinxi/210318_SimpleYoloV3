import os

import numpy
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from model import yolov3net
from util import yolo_utils


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
        print("YoloV3 generate Success")

    def detect_image(self, image: Image.Image) -> Image.Image:
        """
        检测图片
        """
        image_shape = numpy.array(numpy.shape(image)[0:2])  # 保存原始的图像大小

        # 给图像增加灰条，实现不失真的resize
        crop_img = numpy.array(
            yolo_utils.letterbox_image(image, (self.config["image_width"], self.config["image_height"])))

        # 输入图像归一化到 0~1
        photo = numpy.array(crop_img, dtype=numpy.float32) / 255.0

        # Image.open()读取的通道顺序是RGB
        # 深度学习中普遍使用BGR而不用RGB
        # RGB -> BGR
        photo = numpy.transpose(photo, (2, 0, 1))

        # 添加上batch_size维度
        images = [photo,photo]

        with torch.no_grad():  # 没有梯度传递，进行图像检测
            images = torch.from_numpy(numpy.asarray(images))  # 输入图片转化为 torch Tensor
            if self.config["cuda"] and torch.cuda.is_available():  # 配置 GPU 且 Cuda 可用
                images = images.cuda()

            # 将图像输入网络当中进行预测
            outputs = self.net(images)
            # 预测框列表
            output_list = []
            # 对三种大小的预测框进行解析
            for i in range(3):  # 顺序：大，中，小
                output_list.append(self.yolov3_decodes[i](outputs[i]))

            for i, o in enumerate(output_list): print("output_list", i, ":", o.shape)
            """
            output_list 0 : torch.Size([1, 507, 85])，13*13*3 = 507
            output_list 1 : torch.Size([1, 2028, 85])，26*26*3=2028
            output_list 2 : torch.Size([1, 8112, 85])，52*52*3=8112
            """

            # 将预测框进行堆叠
            output = torch.cat(output_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            print("output:", output.shape)  # torch.Size([1, 10647, 85])

            # 进行非极大抑制
            batch_detections = yolo_utils.non_max_suppression(
                output,  # 预测框列表
                self.config["classes"],  # 类别数目
                conf_thres=self.config["confidence"],  # 置信度
                nms_thres=self.config["iou"]  # iou 阈值
            )

            # 如果没有检测出物体，返回原图
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # x1, y1, x2, y2, obj_conf, class_conf, class_pred
            print("batch_detections:", batch_detections.shape)  # (12, 7)

            # 对预测框进行得分筛选
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.config["confidence"]

            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]  # 置信度
            top_label = numpy.array(batch_detections[top_index, -1], numpy.int32)  # 标签
            top_boxes = numpy.array(batch_detections[top_index, :4])  # 预测框

            top_xmin, top_ymin, top_xmax, top_ymax = numpy.expand_dims(top_boxes[:, 0], -1), \
                                                     numpy.expand_dims(top_boxes[:, 1], -1), \
                                                     numpy.expand_dims(top_boxes[:, 2], -1), \
                                                     numpy.expand_dims(top_boxes[:, 3], -1)

            # 从灰条检测图像中恢复原始的检测框
            boxes = yolo_utils.yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                  numpy.array(
                                                      [self.config["image_width"], self.config["image_height"]]),
                                                  image_shape)

        for i, c in enumerate(top_label):
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, numpy.floor(top + 0.5).astype('int32'))
            left = max(0, numpy.floor(left + 0.5).astype('int32'))
            bottom = min(numpy.shape(image)[0], numpy.floor(bottom + 0.5).astype('int32'))
            right = min(numpy.shape(image)[1], numpy.floor(right + 0.5).astype('int32'))

            # 画框框
            draw = ImageDraw.Draw(image)
            print(top, left, bottom, right)

            draw.rectangle([left + i, top + i, right - i, bottom - i])
            del draw

        return image


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    from model import config

    yolov3 = YoloV3(config.Config)

    print("../images/test0.png")

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
