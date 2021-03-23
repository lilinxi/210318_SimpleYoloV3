from __future__ import division

import numpy as np
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


class DecodeBox(nn.Module):
    """
    将预测结果解析为预测框
    """

    def __init__(self, anchors, classes, image_height, image_width, cuda) -> None:
        """
        :param anchors: 解析当前预测层使用的锚框
        :param classes: 总分类数目
        :param image_height: 处理前图像高
        :param image_width: 处理前图像宽
        :param cuda: 是否使用 cuda
        """
        super().__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)  # anchor 数量
        self.classes = classes
        self.bbox_attrs = 4 + 1 + classes  # 每个 anchor 框的属性数目
        self.image_height = image_height
        self.image_width = image_width
        self.cuda = cuda

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """
        (batch_size, 3*(4+1+classes), feature_height, feature_width) ->  (batch_size, predict_box_num = 3 * feature_height * feature_width, 4+1+classes)
        :param input_feature: 解析前的特征层
        :return: 解析后的预测框
        """
        batch_size = input_feature.shape[0]  # 批次大小
        feature_height = input_feature.shape[2]  # 特征图高
        feature_width = input_feature.shape[3]  # 特征图宽

        # 步长 stride = 32、16、8，也即特征层上的一个点相当于处理前图像的多少像素
        stride_height = self.image_height / feature_height
        stride_width = self.image_width / feature_width

        # # 特征层上的 anchor 大小
        # scaled_anchors = [
        #     (
        #         anchor_widthidth / stride_width,
        #         anchor_heighteight / stride_height
        #     )
        #     for anchor_widthidth, anchor_heighteight in self.anchors
        # ]

        # 对预测层的 Tensor 维度进行变换
        # (batch_size, 3*(4+1+classes), feature_height, feature_width) ->  (batch_size, 3, feature_height, feature_width, 4+1+classes)
        prediction = input_feature.view(
            batch_size,  # batch_size
            self.num_anchors,  # 3
            self.bbox_attrs,  # 4+1+classes
            feature_height,  # feature_height
            feature_width  # feature_width
        ).permute(0, 1, 3, 4, 2).contiguous()
        # 如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作。

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  # (batch_size, 3, feature_height, feature_width, x)
        y = torch.sigmoid(prediction[..., 1])  # (batch_size, 3, feature_height, feature_width, y)
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # (batch_size, 3, feature_height, feature_width, w)
        h = prediction[..., 3]  # (batch_size, 3, feature_height, feature_width, h)
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])  # (batch_size, 3, feature_height, feature_width, conf)
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # (batch_size, 3, feature_height, feature_width, classes)

        # 根据是否使用 GPU 改变 Tensor 类型
        FloatTensor: type = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        LongTensor: type = torch.cuda.LongTensor if self.cuda else torch.LongTensor

        # 构造左上角点网格
        # (width) ->
        # (height, 1) * (width) = (height, width) ->
        # (batch_size * self.num_anchors, 1, 1) * (height, width) ->
        # (batch_size * self.num_anchors, height, width) ->
        # view: (batch_size, self.num_anchors, height, width)
        grid_x = torch.linspace(0, feature_width - 1, feature_width) \
            .repeat(feature_height, 1) \
            .repeat(batch_size * self.num_anchors, 1, 1) \
            .view(x.shape).type(FloatTensor)
        # (height) ->
        # (width, 1) * (height) = (width, height) ->
        # transpose: (height, width)
        # (batch_size * self.num_anchors, 1, 1) * (height, width) ->
        # (batch_size * self.num_anchors, height, width) ->
        # view: (batch_size, self.num_anchors, height, width)
        grid_y = torch.linspace(0, feature_height - 1, feature_height) \
            .repeat(feature_width, 1) \
            .t() \
            .repeat(batch_size * self.num_anchors, 1, 1) \
            .view(y.shape).type(FloatTensor)

        # 构造先验框宽高网格
        # (3, 2) -> torch.Size([3, 1])
        anchor_width = FloatTensor(self.anchors).index_select(1, LongTensor([0]))  # 获取所有 anchor 的宽，即第一维度上索引为零的值
        anchor_height = FloatTensor(self.anchors).index_select(1, LongTensor([1]))  # 获取所有 anchor 的高，即第一维度上索引为一的值
        # (3, 1) ->
        # (batch_size, 1) * (3, 1) = (batch_size * 3, 1) -> 
        # (batch_size * 3, 1) * (1, 1, feature_height * feature_width) = (batch_size * 3, 1, feature_height * feature_width) ->
        # view: (batch_size, 3, feature_height, feature_width)
        anchor_width = anchor_width.repeat(batch_size, 1).repeat(1, 1, feature_height * feature_width).view(w.shape)
        anchor_height = anchor_height.repeat(batch_size, 1).repeat(1, 1, feature_height * feature_width).view(h.shape)

        # 利用预测结果调节先验框的中心和宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # 创建解析后的预测框
        pred_boxes[..., 0] = (x.detach() + grid_x) * stride_width  # 预测框为先验框中心加偏移
        pred_boxes[..., 1] = (y.detach() + grid_y) * stride_height  # 预测框为先验框中心加偏移
        pred_boxes[..., 2] = torch.exp(w.detach()) * anchor_width  # 预测框为当前特征层的先验框大小乘以预测系数
        pred_boxes[..., 3] = torch.exp(h.detach()) * anchor_height  # 预测框为当前特征层的先验框大小乘以预测系数

        # 等同于
        # _scale = torch.Tensor([stride_width, stride_height] * 2).type(FloatTensor)
        # pred_boxes = pred_boxes * _scale

        # -1 表示拼接最后一个维度
        # pred_boxes: (batch_size, 3, feature_height, feature_width, 4) -> (batch_size, predict_box_num, 4) +
        # conf: (batch_size, 3, feature_height, feature_width, 1) -> (batch_size, predict_box_num, 1) +
        # pred_cls: (batch_size, 3, feature_height, feature_width, classes) -> (batch_size, predict_box_num, classes) =
        # (batch_size, predict_box_num, 4+1+classes)
        predict_bbox_attrs = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.classes)
            ), -1)

        return predict_bbox_attrs.detach()


def letterbox_image(image: Image.Image, scale_width: int, scale_height: int) -> Image.Image:
    """
    检测图像增加灰条，实现不失真的图像等比例放缩

    :param image: 原始图像
    :param scale_width: 目标图像宽度
    :param scale_height: 目标图像高度
    :return:
    """
    image_width, image_height = image.size

    scale = min(scale_width / image_width, scale_height / image_height)  # 图像放缩倍数，取最小的那个放缩值

    nw = int(image_width * scale)  # 放缩后的图像大小
    nh = int(image_height * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 图像放缩，等比例

    new_image = Image.new('RGB', (scale_width, scale_height), (128, 128, 128))  # 创建一张灰色底板作为返回的图像
    new_image.paste(image, ((scale_width - nw) // 2, (scale_height - nh) // 2))  # 等比例放缩后的图像粘贴到底板中央

    return new_image


def non_max_suppression(
        prediction: torch.Tensor, classes: int,
        conf_threshold: float = 0.5, nms_iou_threshold: float = 0.4) -> List[torch.Tensor]:
    """
    进行非极大值抑制，并将预测结果的格式转换成左上角右下角的格式。

    预测结果格式转换
    置信度筛选
    nms 筛选

    :param prediction: 预测框列表，torch.Size([1, 10647, 85]) = (batch_size, 13*13*3 + 26*26*3 + 52*52*3 = 10647, 3*(x+y+w+h+conf+classes))
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


def yolo_correct_boxes(
        xmin, ymin, xmax, ymax,
        image_input_width, image_input_height,
        image_raw_width, image_raw_height):
    """
    从灰条检测图像中恢复原始的检测框
    """
    input_shape = np.asarray([image_input_width, image_input_height])
    image_shape = np.asarray([image_raw_width, image_raw_height])

    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((ymin + ymax) / 2, (xmin + xmax) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((ymax - ymin, xmax - xmin), axis=-1) / input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)

    return boxes
