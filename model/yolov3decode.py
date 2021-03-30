from __future__ import division

import numpy
from PIL import Image

import torch
import torch.nn as nn
import torchvision.ops

# -----------------------------------------------------------------------------------------------------------#
# class DecodeBox(nn.Module) # 将预测结果解析为预测框
#
# def letterbox_image(image: Image.Image, scale_width: int, scale_height: int) -> Image.Image:
#    检测图像增加灰条，实现不失真的图像等比例放缩
#
# def non_max_suppression(
#         prediction: torch.Tensor, classes: int,
#         conf_threshold: float = 0.5, nms_iou_threshold: float = 0.4) -> List[torch.Tensor]:
#     进行非极大值抑制，并将预测结果的格式转换成左上角右下角的格式。
#
# def yolo_correct_boxes(
#         xmin, ymin, xmax, ymax,
#         image_input_width, image_input_height,
#         image_raw_width, image_raw_height):
#     从灰条检测图像中恢复原始的检测框
# -----------------------------------------------------------------------------------------------------------#
from typing import List


class YoloV3Decode(nn.Module):
    """
    将预测结果解析为预测框
    """

    def __init__(self, config: dict) -> None:
        """
        :param anchors: 解析当前预测层使用的锚框
        :param classes: 总分类数目
        :param image_height: 处理前图像高
        :param image_width: 处理前图像宽
        :param cuda: 是否使用 cuda
        """
        super().__init__()

        self.anchors = config["anchors"]
        self.anchors_13 = self.anchors[0]
        self.anchors_26 = self.anchors[1]
        self.anchors_52 = self.anchors[2]
        # self.cur_anchors = None
        # self.cur_anchors_num = None

        self.classes = config["classes"]
        self.bbox_attrs = 4 + 1 + self.classes

        self.image_height = config["image_height"]
        self.image_width = config["image_width"]

        self.cuda = config["cuda"]

    def forward(self, predict_feature: torch.Tensor) -> torch.Tensor:
        """
        :param predict_feature: 解析前的特征层
        :return: 解析后的预测框
        """
        # 1. 解析预测网络输出的特征层的各维度属性
        batch_size = predict_feature.shape[0]
        predict_feature_height = predict_feature.shape[2]
        predict_feature_width = predict_feature.shape[3]
        assert predict_feature_height == predict_feature_width
        assert predict_feature_height in [13, 26, 52]

        # 2. 计算当前特征层的步长
        stride_height = self.image_height / predict_feature_height
        stride_width = self.image_width / predict_feature_width

        # 3. 确定当前特征层的 anchors
        if predict_feature_height == 13:
            cur_anchors = self.anchors_13
        elif predict_feature_height == 26:
            cur_anchors = self.anchors_26
        elif predict_feature_height == 52:
            cur_anchors = self.anchors_52
        else:
            raise Exception("unexpected error")

        cur_anchors_num = len(cur_anchors)
        assert cur_anchors_num * self.bbox_attrs == predict_feature.shape[1]

        # 4. 将预测网络输出的特征层进行维度变换，将预测框个数与预测属性分开，并将预测属性转置为末位维度的属性，便于提取和解析
        predict_feature = predict_feature.contiguous().view(
            batch_size,
            cur_anchors_num,
            self.bbox_attrs,
            predict_feature_height,
            predict_feature_width,
        ).permute(0, 1, 3, 4, 2).contiguous()

        # 5. 分隔预测属性
        predict_x = predict_feature[..., 0]
        predict_y = predict_feature[..., 1]
        predict_w = predict_feature[..., 2]
        predict_h = predict_feature[..., 3]
        predict_obj_conf = predict_feature[..., 4]
        predict_class_conf_list = predict_feature[..., 5:]

        # 6. 解析 xy
        norm_predict_x = torch.sigmoid(predict_x)
        norm_predict_y = torch.sigmoid(predict_y)
        # 6.1 构造 grid tensor
        grid_x = torch.linspace(0, predict_feature_width - 1, predict_feature_width) \
            .repeat(predict_feature_height, 1) \
            .repeat(batch_size * cur_anchors_num, 1, 1) \
            .view(predict_x.shape)
        grid_y = torch.linspace(0, predict_feature_height - 1, predict_feature_height) \
            .repeat(predict_feature_width, 1) \
            .t() \
            .repeat(batch_size * cur_anchors_num, 1, 1) \
            .view(predict_y.shape)
        # 6.2 叠加 grid tensor
        grid_predict_x = norm_predict_x + grid_x
        grid_predict_y = norm_predict_y + grid_y
        # 6.3 乘以步长
        strided_predict_x = grid_predict_x * stride_width
        strided_predict_y = grid_predict_y * stride_height

        # 7. 解析 wh
        # 7.1 构造 anchor tensor
        anchor_width = torch.unsqueeze(torch.Tensor(cur_anchors)[:, 0], dim=1)
        anchor_height = torch.unsqueeze(torch.Tensor(cur_anchors)[:, 1], dim=1)
        grid_anchor_width = anchor_width.repeat(batch_size, 1). \
            repeat(1, 1, predict_feature_height * predict_feature_width). \
            view(predict_w.shape)
        grid_anchor_height = anchor_height.repeat(batch_size, 1). \
            repeat(1, 1, predict_feature_height * predict_feature_width). \
            view(predict_h.shape)
        # 7.2 乘以 anchor tensor
        anchord_width = torch.exp(predict_w) * grid_anchor_width
        anchord_height = torch.exp(predict_h) * grid_anchor_height

        # 8. 解析 obj_conf
        norm_predict_obj_conf = torch.sigmoid(predict_obj_conf)

        # 9. 解析 class_conf_list
        norm_predict_class_conf_list = torch.sigmoid(predict_class_conf_list)

        # 10. 拼接解析结果
        predict_bbox_attrs = torch.cat(
            (
                strided_predict_x.view(batch_size, -1, 1),
                strided_predict_y.view(batch_size, -1, 1),
                anchord_width.view(batch_size, -1, 1),
                anchord_height.view(batch_size, -1, 1),
                norm_predict_obj_conf.view(batch_size, -1, 1),
                norm_predict_class_conf_list.view(batch_size, -1, self.classes)
            ), -1)

        return predict_bbox_attrs


def non_max_suppression(
        prediction: torch.Tensor, classes: int,
        conf_threshold: float = 0.5, nms_iou_threshold: float = 0.4) -> List[torch.Tensor]:
    """
    进行非极大值抑制，并将预测结果的格式转换成左上角右下角的格式。

    预测结果格式转换
    置信度筛选
    nms 筛选

    :param prediction: 预测框列表，torch.Size([1, 10647, 85]) = (batch_size, 13*13*3 + 26*26*3 + 52*52*3 = 10647, (x+y+w+h+conf+classes))
    :param classes: 类别数目
    :param conf_threshold: 置信度阈值
    :param nms_iou_threshold: iou 阈值
    :return:
        prediction_after_nms: batch_size * (box_num, xmin + ymin + xmax + ymax + obj_conf + class_conf + class_label = 7)
    """

    # 复制原预测框列表，将预测框格式改为左上角右下角的格式
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x - w/2 = xmin
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y - h/2 = ymin
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x + w/2 = xmax
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y + h/2 = ymax

    # 将新格式替换原预测框列表
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 创建返回结果列表，len 为 batch_size
    batch_size = len(prediction)
    prediction_after_nms = [None] * batch_size

    for image_index, image_prediction in enumerate(prediction):  # 遍历每张图片和预测数据
        # 对种类预测部分取max。
        image_classes_prediction = image_prediction[:, 5:5 + classes]
        class_conf, class_label = torch.max(image_classes_prediction, 1, keepdim=True)
        # class_conf: 预测种类得分的最大值作为置信度
        # class_label: 预测种类得分的最大值的索引作为预测标签

        # 利用置信度进行第一轮筛选
        # (10647, 1) * (10647, 1) -> (10647, 1) -> (10647)
        # obj_conf * class_conf >= conf_threshold
        conf_mask = (image_prediction[:, 4] * class_conf[:, 0] >= conf_threshold).squeeze()

        # 根据置信度进行预测结果的筛选，只保留一小部分预测框
        image_prediction_after_conf_threshold = image_prediction[
            conf_mask]  # torch.Size([10647, 85]) -> (threshold_size, 85)
        class_conf_after_conf_threshold = class_conf[conf_mask]  # torch.Size([10647, 1]) -> (threshold_size, 1)
        class_label_after_conf_threshold = class_label[conf_mask]  # torch.Size([10647, 1]) -> (threshold_size, 1)

        if not image_prediction_after_conf_threshold.size(0):  # 如果没有预测框了则开始下一个图片，即 threshold_size = 0
            continue

        # -------------------------------------------------------------------------#
        #   创建 detections  [threshold_size, 7]
        #   7 的内容为：xmin + ymin + xmax + ymax + obj_conf + class_conf + class_label
        # -------------------------------------------------------------------------#
        detections = torch.cat(
            (
                image_prediction_after_conf_threshold[:, :5],
                class_conf_after_conf_threshold.float(),
                class_label_after_conf_threshold.float()
            ), 1)

        #  获得预测结果中包含的所有种类
        unique_labels = detections[:, -1].unique()

        # 如果使用 GPU 则转化为 GPU 存储
        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # 遍历每一个类别
        for label in unique_labels:
            # 获得某一类得分筛选后全部的预测结果
            detections_in_label = detections[detections[:, -1] == label]

            #  使用官方自带的非极大抑制会速度更快一些！
            keep = torchvision.ops.nms(
                detections_in_label[:, :4],  # 预测框
                detections_in_label[:, 4] * detections_in_label[:, 5],  # 置信度：obj_conf * classes_conf
                nms_iou_threshold  # iou 阈值
            )

            # 筛选之后的预测结果
            max_detections_in_label = detections_in_label[keep]

            # 添加筛选之后的预测结果，直接添加或者拼接在后面
            if prediction_after_nms[image_index] is None:
                prediction_after_nms[image_index] = max_detections_in_label
            else:
                prediction_after_nms[image_index] = torch.cat(
                    (prediction_after_nms[image_index], max_detections_in_label), 0)

    return prediction_after_nms
