from __future__ import division

import torch
import torch.nn as nn
import torchvision.ops

from typing import List


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


class YoloV3Decode(nn.Module):
    """
    将预测结果解析为预测框
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.anchors = config["anchors"]
        self.anchors_13 = self.anchors[0]
        self.anchors_26 = self.anchors[1]
        self.anchors_52 = self.anchors[2]

        self.classes = config["classes"]
        self.bbox_attrs = 4 + 1 + self.classes

        self.image_height = config["image_height"]
        self.image_width = config["image_width"]

        self.cuda = config["cuda"]

    def forward(self, predict_feature: torch.Tensor) -> torch.Tensor:
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
        # 6.4 改变维度
        viewd_predict_x = strided_predict_x.view(batch_size, -1, 1)
        viewd_predict_y = strided_predict_y.view(batch_size, -1, 1)

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
        anchord_predict_width = torch.exp(predict_w) * grid_anchor_width
        anchord_predict_height = torch.exp(predict_h) * grid_anchor_height
        # 7.3 改变维度
        viewd_anchord_predict_width = anchord_predict_width.view(batch_size, -1, 1)
        viewd_anchord_predict_height = anchord_predict_height.view(batch_size, -1, 1)
        # 7.4 除以 2
        half_viewd_anchord_predict_width = torch.mul(viewd_anchord_predict_width, 0.5)
        half_viewd_anchord_predict_height = torch.mul(viewd_anchord_predict_height, 0.5)

        # 8. 解析 obj_conf
        norm_predict_obj_conf = torch.sigmoid(predict_obj_conf)
        # 8.1 改变维度
        viewd_predict_obj_conf = norm_predict_obj_conf.view(batch_size, -1, 1)

        # 9. 解析 class_conf_list
        norm_predict_class_conf_list = torch.sigmoid(predict_class_conf_list)
        # 9.1 改变维度
        viewd_predict_class_conf_list = norm_predict_class_conf_list.view(batch_size, -1, self.classes)
        # 9.2 预测种类得分的最大值作为置信度, 预测种类得分的最大值的索引作为预测标签
        norm_predict_class_conf, predict_class_label = torch.max(viewd_predict_class_conf_list, 2, keepdim=True)

        # 10. 拼接解析结果，拼接最后一个维度
        predict_bbox_attrs = torch.cat(
            (
                viewd_predict_x - half_viewd_anchord_predict_width,
                viewd_predict_y - half_viewd_anchord_predict_height,
                viewd_predict_x + half_viewd_anchord_predict_width,
                viewd_predict_y + half_viewd_anchord_predict_height,
                viewd_predict_obj_conf * norm_predict_class_conf,
                predict_class_label,
            ), -1)

        return predict_bbox_attrs


def non_max_suppression(
        predict_bbox_attrs: torch.Tensor,
        conf_threshold: float,
        nms_iou_threshold: float
) -> List[torch.Tensor]:
    """
    进行非极大值抑制

    置信度筛选
    nms 筛选

    :param prediction: 预测框列表，torch.Size([1, 10647, 85]) = (batch_size, 13*13*3 + 26*26*3 + 52*52*3 = 10647, (x+y+w+h+conf+classes))
    :param classes: 类别数目
    :param conf_threshold: 置信度阈值
    :param nms_iou_threshold: iou 阈值
    :return:
        prediction_after_nms: batch_size * (box_num, xmin + ymin + xmax + ymax + obj_conf + class_conf + class_label = 7)
    """

    # 1. 返回值为长度为 batch_size 的列表，每张图片做单独的过滤处理
    predict_bbox_attrs_after_nms = [None] * predict_bbox_attrs.shape[0]

    # 2. 遍历每张图片和预测数据
    for image_index, image_predict_bbox_attrs in enumerate(predict_bbox_attrs):
        # 3. 置信度筛选
        conf = image_predict_bbox_attrs[:, 4]
        conf_mask = conf >= conf_threshold
        image_predict_after_conf_threshold = image_predict_bbox_attrs[conf_mask]

        # 4. 如果没有预测框了则开始下一个图片，即 threshold_size = 0
        if not image_predict_after_conf_threshold.shape[0]:
            continue

        # 5. 获得预测结果中包含的所有种类，对每个种类进行 iou 筛选
        unique_labels = image_predict_after_conf_threshold[:, 5].unique()
        for label in unique_labels:
            # 6. 获得某个种类的所有预测框
            image_predict_in_label = image_predict_after_conf_threshold[
                image_predict_after_conf_threshold[:, 5] == label
                ]

            #  使用官方自带的非极大抑制会速度更快一些！
            keep = torchvision.ops.nms(
                image_predict_in_label[:, :4],  # 预测框
                image_predict_in_label[:, 4] ,#* image_predict_in_label[:, 5],  # 置信度：obj_conf * classes_conf
                nms_iou_threshold  # iou 阈值
            )

            # 筛选之后的预测结果
            max_detections_in_label = image_predict_in_label[keep]

            # 添加筛选之后的预测结果，直接添加或者拼接在后面
            if predict_bbox_attrs_after_nms[image_index] is None:
                predict_bbox_attrs_after_nms[image_index] = max_detections_in_label
            else:
                predict_bbox_attrs_after_nms[image_index] = torch.cat(
                    (predict_bbox_attrs_after_nms[image_index], max_detections_in_label), 0)

    return predict_bbox_attrs_after_nms
