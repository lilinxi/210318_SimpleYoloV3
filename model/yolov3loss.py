from typing import List

import numpy

import torch


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height


def jaccard_tensor(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,4]
    Return:
        jaccard overlap: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class YoloV3Loss(torch.nn.Module):
    """
    YoloV3 损失函数
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.lambda_xy = 1.0  # 预测框中心误差权重
        self.lambda_wh = 1.0  # 预测框大小误差权重
        self.lambda_noobj = 1.0  # 预测框置信度误差权重
        self.lambda_obj = 1.0  # 预测框置信度误差权重
        self.lambda_class = 1.0  # 预测框类别误差权重
        self.lambda_conf = 1.0  # 预测框类别误差权重

        self.normd_anchors = numpy.asarray(config["anchors"]).astype(numpy.float)
        self.normd_anchors[:, :, 0] /= config["image_width"]
        self.normd_anchors[:, :, 1] /= config["image_height"]
        self.normd_anchors = self.normd_anchors.reshape((9, 2))
        self.normd_anchors_box = torch.cat(
            (
                torch.zeros((self.normd_anchors.shape[0], 2)),
                torch.from_numpy(self.normd_anchors)
            ), 1)

        self.classes = config["classes"]
        self.bbox_attrs = 4 + 1 + self.classes

        self.ignore_threshold = 0.5  # iou 忽略的阈值

    def pyramid_target(self, tensord_target_list: List[torch.Tensor]):
        pyramid_target_list_13 = []
        pyramid_target_list_26 = []
        pyramid_target_list_52 = []

        for tensord_target in tensord_target_list:
            tensord_target_13 = []
            tensord_target_26 = []
            tensord_target_52 = []

            target_box = tensord_target[:, :4].clone().detach()
            target_box[:, 0] = 0
            target_box[:, 1] = 0
            normd_anch_ious = jaccard_tensor(target_box, self.normd_anchors_box)
            max_anch_ious_index = torch.argmax(normd_anch_ious, dim=-1)

            for box_index, anch_index in enumerate(max_anch_ious_index):
                if anch_index in [0, 1, 2]:
                    tensord_target_13.append(tensord_target[box_index].numpy())
                elif anch_index in [3, 4, 5]:
                    tensord_target_26.append(tensord_target[box_index].numpy())
                elif anch_index in [6, 7, 8]:
                    tensord_target_52.append(tensord_target[box_index].numpy())
                else:
                    raise Exception("unexpected error")

            pyramid_target_list_13.append(torch.as_tensor(tensord_target_13))
            pyramid_target_list_26.append(torch.as_tensor(tensord_target_26))
            pyramid_target_list_52.append(torch.as_tensor(tensord_target_52))

        return pyramid_target_list_13, pyramid_target_list_26, pyramid_target_list_52

    def decode_pyramid_target(self,
                              pyramid_target_list: List[torch.Tensor],
                              pyramid_normd_anchors: numpy.ndarray,
                              pyramid_features: int
                              ) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor
    ):
        assert pyramid_features in [13, 26, 52]

        if pyramid_features == 13:
            pyramid_anch_index_list = [0, 1, 2]
        elif pyramid_features == 26:
            pyramid_anch_index_list = [3, 4, 5]
        elif pyramid_features == 52:
            pyramid_anch_index_list = [6, 7, 8]
        else:
            raise Exception("unexpected error")

        batch_size = len(pyramid_target_list)

        target_x = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        target_y = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        target_w = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        target_h = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)

        target_loss_weight_xw = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        target_loss_weight_yh = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)

        target_obj_conf = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        target_class_conf_list = torch.zeros(batch_size, 3, pyramid_features, pyramid_features,
                                             self.classes)

        target_obj_mask = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        target_noobj_mask = torch.ones(batch_size, 3, pyramid_features, pyramid_features)

        # 遍历这一个批次所有的图片
        for bs_i, pyramid_target in enumerate(pyramid_target_list):
            if pyramid_target.shape[0] == 0:
                continue

            truth_feature_box = pyramid_target[:, 0:4] * pyramid_features

            truth_grid_x = torch.floor(truth_feature_box[:, 0]).int()
            truth_grid_y = torch.floor(truth_feature_box[:, 1]).int()

            truth_x = truth_feature_box[:, 0] - truth_grid_x
            truth_y = truth_feature_box[:, 1] - truth_grid_y

            target_box = pyramid_target[:, :4].clone().detach()
            target_box[:, 0] = 0
            target_box[:, 1] = 0
            normd_anch_ious = jaccard_tensor(target_box, self.normd_anchors_box)
            max_anch_ious_index = torch.argmax(normd_anch_ious, dim=-1)

            for box_i, anch_i in enumerate(max_anch_ious_index):
                if anch_i not in pyramid_anch_index_list:
                    continue
                pyramid_anch_i = anch_i % 3

                truth_w_box = torch.log(pyramid_target[box_i][2] / self.normd_anchors[anch_i][0])
                truth_h_box = torch.log(pyramid_target[box_i][3] / self.normd_anchors[anch_i][1])

                target_x[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_x[box_i]
                target_y[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_y[box_i]

                target_w[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_w_box
                target_h[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_h_box

                target_loss_weight_xw[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = \
                    pyramid_target[box_i][2]
                target_loss_weight_yh[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = \
                    pyramid_target[box_i][3]

                target_obj_conf[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = 1
                target_class_conf_list[
                    bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i], pyramid_target[box_i][4].int()] = 1

                target_obj_mask[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = 1
                target_noobj_mask[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = 0

        return target_x, target_y, target_w, target_h, target_loss_weight_xw, target_loss_weight_yh, target_obj_conf, target_class_conf_list, target_obj_mask, target_noobj_mask

    def compute_loss(self, predict_feature: torch.Tensor, decoded_target) -> (
            torch.Tensor, torch.Tensor):
        (target_x, target_y, target_w, target_h, target_loss_weight_xw, target_loss_weight_yh, target_obj_conf,
         target_class_conf_list, target_obj_mask, target_noobj_mask) = decoded_target

        predict_feature = predict_feature.view(
            predict_feature.shape[0],
            3,
            self.bbox_attrs,
            predict_feature.shape[2],
            predict_feature.shape[3],
        ).permute(0, 1, 3, 4, 2).contiguous()

        predict_x = torch.sigmoid(predict_feature[..., 0])
        predict_y = torch.sigmoid(predict_feature[..., 1])
        predict_w = predict_feature[..., 2]
        predict_h = predict_feature[..., 3]
        predict_obj_conf = torch.sigmoid(predict_feature[..., 4])
        predict_class_conf_list = torch.sigmoid(predict_feature[..., 5:])

        target_loss_scale = 2 - target_loss_weight_xw * target_loss_weight_yh

        loss_x = torch.sum(torch.nn.BCELoss()(predict_x, target_x) * target_loss_scale * target_obj_mask)
        loss_y = torch.sum(torch.nn.BCELoss()(predict_y, target_y) * target_loss_scale * target_obj_mask)

        loss_w = torch.sum(torch.nn.MSELoss()(predict_w, target_w) * 0.5 * target_loss_scale * target_obj_mask)
        loss_h = torch.sum(torch.nn.MSELoss()(predict_h, target_h) * 0.5 * target_loss_scale * target_obj_mask)

        loss_conf = self.lambda_obj * torch.sum(
            torch.nn.BCELoss()(predict_obj_conf, target_obj_mask) * target_obj_mask) + \
                    self.lambda_noobj * torch.sum(
            torch.nn.BCELoss()(predict_obj_conf, target_obj_mask) * target_noobj_mask)

        loss_class = torch.sum(torch.nn.BCELoss()(predict_class_conf_list[target_obj_mask == 1],
                                                  target_class_conf_list[target_obj_mask == 1]))

        print("\n---------------------------------------")
        print(loss_x, loss_y)
        print(loss_w, loss_h)
        print(loss_conf, loss_class)
        print("---------------------------------------\n")

        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_class * self.lambda_class

        return loss, torch.sum(target_obj_mask)

    def forward(self, predict_feature_list,
                tensord_target_list: List[torch.Tensor]) -> torch.Tensor:
        # pyramid_target_list_13, pyramid_target_list_26, pyramid_target_list_52 = \
        #     self.pyramid_target(tensord_target_list)
        # pyramid_normd_anchors_13, pyramid_normd_anchors_26, pyramid_normd_anchors_52 = \
        #     self.normd_anchors[0:3], self.normd_anchors[3, 6], self.normd_anchors[6, 9]

        target_13 = self.decode_pyramid_target(tensord_target_list, None, 13)
        target_26 = self.decode_pyramid_target(tensord_target_list, None, 26)
        target_52 = self.decode_pyramid_target(tensord_target_list, None, 52)

        loss_13, loss_13_num = self.compute_loss(predict_feature_list[0], target_13)
        loss_26, loss_26_num = self.compute_loss(predict_feature_list[1], target_26)
        loss_52, loss_52_num = self.compute_loss(predict_feature_list[2], target_52)

        loss_list = []

        if not torch.isnan(loss_13):
            loss_list.append(loss_13)
        if not torch.isnan(loss_26):
            loss_list.append(loss_26)
        if not torch.isnan(loss_52):
            loss_list.append(loss_52)

        assert len(loss_list) is not 0

        loss = sum(loss_list)

        loss_num = loss_13_num + loss_26_num + loss_52_num

        return loss / loss_num
