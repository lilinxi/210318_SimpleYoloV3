import logging
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List


def jaccard(_box_a, _box_b):
    """
    雅卡尔指数（英語：Jaccard index），又称为并交比（Intersection over Union）、雅卡尔相似系数（Jaccard similarity coefficient），是用于比较样本集的相似性
    计算预测框和先验框的交比 iou（x，y，w，h）
    :param _box_a: 预测框，box*4
    :param _box_b: 先验框，9*4
    :return: 交比，box*9
    """

    # 计算真实框的左上角和右下角
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2

    # 计算先验框的左上角和右下角
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)

    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]

    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    # 求IOU
    union = area_a + area_b - inter

    return inter / union  # [A,B]


def clip_by_tensor(t, t_min, t_max):
    """
    将 t 限制在 min 和 max 之间
    """
    t = t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max

    return result


def MSELoss(pred, target):
    """
    平方误差
    """

    return (pred - target) ** 2


def BCELoss(pred, target):
    """
    交叉熵误差
    """

    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)  # 预测值限制在 0~1

    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)

    return output


class YoloLoss(nn.Module):
    """
    Yolo 损失函数
    """

    def __init__(self, anchors, num_classes, image_size, cuda, normalize):
        """
        :param anchors: 预测框列表
        :param num_classes: 类别数目
        :param image_size: 图片大小，width * height
        :param cuda: 是否使用 Cuda
        :param normalize: 是否对损失进行归一化
        """
        super(YoloLoss, self).__init__()  # TODO 三个损失函数的参数都是一样的

        """
        print("[YoloLoss] YoloLoss Init...")
        print("[YoloLoss] anchors:\n", anchors)
        print("[YoloLoss] num_classes:", num_classes)
        print("[YoloLoss] image_size:", image_size)
        print("[YoloLoss] cuda:", cuda)
        print("[YoloLoss] normalize:", normalize)
        """

        """
[YoloLoss] YoloLoss Init...
[YoloLoss] anchors:
 [[116  90]
 [156 198]
 [373 326]
 [ 30  61]
 [ 62  45]
 [ 59 119]
 [ 10  13]
 [ 16  30]
 [ 33  23]]
[YoloLoss] num_classes: 80
[YoloLoss] image_size: (416, 416)
[YoloLoss] cuda: False
[YoloLoss] normalize: True
        """

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 4 + 1 + num_classes  # 每个预测框有 4+1+classes 个属性

        # 计算特征层的宽高：416/13=32，416/26=16，416/52=8
        self.feature_length = [image_size[0] // 32, image_size[0] // 16, image_size[0] // 8]  # [13, 26, 52]
        self.image_width = image_size[0]  # 416 * 416
        self.image_height = image_size[1]  # 416 * 416

        self.ignore_threshold = 0.5  # iou 忽略的阈值

        self.lambda_xy = 1.0  # 预测框中心误差权重
        self.lambda_wh = 1.0  # 预测框大小误差权重
        self.lambda_conf = 1.0  # 预测框置信度误差权重
        self.lambda_cls = 1.0  # 预测框类别误差权重

        self.cuda = cuda and torch.cuda.is_available()  # Cuda 可用且设置了使用 Cuda
        self.normalize = normalize

    def forward(self, output: torch.Tensor, targets: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        计算损失
        :param output: YoloNet 的输出结果
                        （batch_size, 3*(5+num_classes), feature_w, feature_h）
                        torch.Size([16, 255, 13, 13]) or torch.Size([16, 255, 26, 26]) or torch.Size([16, 255, 52, 52])
        :param targets: 检测框真值
        :return:
        """

        batch_size = output.size()[0]  # 一个批次一共多少张图片
        output_feature_width = output.size()[2]  # 特征层的宽
        output_feature_height = output.size()[3]  # 特征层的高

        # -----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_width = stride_height = 32、16、8
        # -----------------------------------------------------------------------#
        stride_width = self.image_width / output_feature_width
        stride_height = self.image_height / output_feature_height

        # 此时获得的 scaled_anchors 大小是相对于特征层的
        scaled_anchors = [
            (
                anchor_width / stride_width,
                anchor_height / stride_height
            )
            for anchor_width, anchor_height in self.anchors
        ]

        """
        print("scaled_anchors:\n", scaled_anchors)
         [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875), # 有效
         (0.9375, 1.90625), (1.9375, 1.40625), (1.84375, 3.71875), # 不属于当前特征层
         (0.3125, 0.40625), (0.5, 0.9375), (1.03125, 0.71875)] # 不属于当前特征层
        """

        # -----------------------------------------------#
        #   输入的 output 一共有三个，他们的 shape 分别是
        #   batch_size, 3*85, 13, 13 -> batch_size, 3, 13, 13, 85
        #   batch_size, 3*85, 26, 26 -> batch_size, 3, 26, 26, 85
        #   batch_size, 3*85, 52, 52 -> batch_size, 3, 52, 52, 85
        # -----------------------------------------------#
        prediction = output.view(  # 对预测值 Tensor 的维度进行置换
            batch_size,
            int(self.num_anchors / 3),  # 每个特征点的 anchor 数目
            self.bbox_attrs,
            output_feature_width, output_feature_height
        ).permute(0, 1, 3, 4, 2).contiguous()
        # 如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作。

        # -----------------------------------------------#
        #   对预测层进行解析，拆分 85 个预测维度为 1，1，1，1，1，80
        #   scaled_anchors * （4 + 1 + classes），对 coco，3*(4+1+20)=75
        #   scaled_anchors * （4 + 1 + classes），对 voc，3*(4+1+80)=255
        # -----------------------------------------------#

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  # torch.Size([16, 3, 13, 13])
        y = torch.sigmoid(prediction[..., 1])  # torch.Size([16, 3, 13, 13])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # torch.Size([16, 3, 13, 13])
        h = prediction[..., 3]  # torch.Size([16, 3, 13, 13])
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])  # torch.Size([16, 3, 13, 13])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # torch.Size([16, 3, 13, 13, 80]) or torch.Size([16, 3, 13, 13, 1])

        """
        print(x.shape)
        print(y.shape)
        print(w.shape)
        print(h.shape)
        print(conf.shape)
        print(pred_cls.shape)
        """

        # ---------------------------------------------------------------#
        #   解析标注的真值数据

        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #   mask        batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   有目标的特征点
        #   noobj_mask  batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   无目标的特征点
        #   tx          batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   中心x偏移情况
        #   ty          batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   中心y偏移情况
        #   tw          batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   宽高调整参数的真实值
        #   th          batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   宽高调整参数的真实值
        #   tconf       batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   置信度真实值
        #   tcls        batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height, num_classes  种类真实值
        #   box_loss_scale_x       batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   宽度缩放比例
        #   box_loss_scale_y       batch_size, num_anchors_per_feature_point = 3, output_feature_width, output_feature_height   高度缩放比例
        # ----------------------------------------------------------------#
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y = \
            self.get_target(
                targets,
                scaled_anchors,
                output_feature_width,
                output_feature_height,
                stride_width,
                stride_height,
                self.ignore_threshold
            )

        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #   忽略预测很准的预测框的损失计算
        # ----------------------------------------------------------------#
        noobj_mask = self.get_ignore(prediction, targets, scaled_anchors,
                                     output_feature_height, output_feature_width,
                                     noobj_mask)

        # cuda 转换
        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()

        # 大小比例
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        # print(box_loss_scale)

        # 计算中心偏移情况的loss，使用BCELoss效果好一些
        loss_x = torch.sum(BCELoss(x, tx) / batch_size * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) / batch_size * box_loss_scale * mask)

        # loss_x = torch.sum(BCELoss(x, tx) / batch_size * mask)
        # loss_y = torch.sum(BCELoss(y, ty) / batch_size * mask)
        # 计算宽高调整值的loss
        loss_w = torch.sum(MSELoss(w, tw) / batch_size * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) / batch_size * 0.5 * box_loss_scale * mask)

        # loss_w = torch.sum(MSELoss(w, tw) / batch_size * 0.5 * mask)
        # loss_h = torch.sum(MSELoss(h, th) / batch_size * 0.5 * mask)
        # 计算置信度的loss
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / batch_size) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask / batch_size)

        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]) / batch_size)

        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

        if self.normalize:  # 损失的数量？
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        else:
            num_pos = batch_size

        """
        print("loss:", type(loss))  # <class 'torch.Tensor'>
        print("loss.shape:", loss.shape)  # torch.Size([])
        print("num_pos:", num_pos)  # tensor(20.)
        """

        # print("loss_x:", loss_x)
        # print("loss_y:", loss_y)
        # print("loss_w:", loss_w)
        # print("loss_h:", loss_h)
        # print("loss_conf:", loss_conf)
        # print("loss_cls:", loss_cls)
        # print("loss:", loss)
        # print("num_pos:", num_pos)
        # exit(-1)
        # tensor(1.3999, grad_fn= < SumBackward0 >)
        # tensor(1.3288, grad_fn= < SumBackward0 >)
        # tensor(0.0318, grad_fn= < SumBackward0 >)
        # tensor(0.0051, grad_fn= < SumBackward0 >)
        # tensor(4.6013, grad_fn= < AddBackward0 >)
        # tensor(0.0004, grad_fn= < SumBackward0 >)
        # tensor(7.3673, grad_fn= < AddBackward0 >)

        return loss, num_pos

    def get_target(self,
                   target: List[torch.Tensor], scaled_anchors: list,
                   output_feature_width: int, output_feature_height: int,
                   stride_width: int, stride_height: int,
                   ignore_threshold: float) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor
    ):
        """
        将目标预测框转化为网络预测的掩码的形式
        :param target: 目标标注，为标注框的数组集合，batch_size * len(box) * 5（中心点 + 宽高 + 类别），中心点和宽高都是 0~1 的
        :param scaled_anchors: scaled_anchors 大小是相对于特征层的
        :param output_feature_width: 特征层宽
        :param output_feature_height: 特征层高
        :param ignore_threshold: iou 忽略的阈值
        :return:
        """

        # 计算一共有多少张图片
        batch_size = len(target)

        # [13, 26, 52] 中根据特征层宽获取特征层的编号，获得当前特征层先验框所属的编号，方便后面对先验框筛选
        anchor_index = [
            [0, 1, 2],  # 大特征层前三个
            [3, 4, 5],  # 中特征层
            [6, 7, 8]  # 小特征层
        ][self.feature_length.index(output_feature_width)]  # feature_length: [13, 26, 52]
        # 先验框开始的坐标
        subtract_index = [0, 3, 6][self.feature_length.index(output_feature_width)]

        # output1：有物体掩码，默认为 0，表示无物体
        mask = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                           requires_grad=False)
        # output2：无物体掩码，默认为 1，表示无物体
        noobj_mask = torch.ones(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                                requires_grad=False)

        # output3，4，5，6：x,y,w,h Tensor，默认为 0
        tx = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                         requires_grad=False)
        ty = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                         requires_grad=False)
        tw = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                         requires_grad=False)
        th = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                         requires_grad=False)

        # output7，8：置信度,类别 Tensor，默认为 0
        tconf = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                            requires_grad=False)
        tcls = torch.zeros(batch_size, int(self.num_anchors / 3), output_feature_width, output_feature_height,
                           self.num_classes,
                           requires_grad=False)

        # output9，10：预测框损失 x，y
        box_loss_scale_x = torch.zeros(batch_size, int(self.num_anchors / 3),
                                       output_feature_width, output_feature_height,
                                       requires_grad=False)
        box_loss_scale_y = torch.zeros(batch_size, int(self.num_anchors / 3),
                                       output_feature_width, output_feature_height,
                                       requires_grad=False)

        # 遍历这一个批次所有的图片
        for i in range(batch_size):
            if len(target[i]) == 0:  # 真值中没有预测框则跳过
                continue

            """
            print(target[i].shape) # torch.Size([2, 5])
            print(target[i])
            tensor([[71.0000, 124.5000, 142.0000, 249.0000, 1.0000],
                    [57.5000, 157.5000, 115.0000, 315.0000, 1.0000]])
            """

            # 计算出正样本在特征层上的中心点
            gxs = target[i][:, 0:1] / stride_width
            gys = target[i][:, 1:2] / stride_height

            """
            print(gxs.shape) # torch.Size([2, 1])
            print(gxs)
            print(gys)
            tensor([[2.2188],
                   [1.7969]])
            tensor([[3.8906],
                    [4.9219]])
            """

            # 计算出正样本相对于特征层的宽高
            gws = target[i][:, 2:3] / stride_width
            ghs = target[i][:, 3:4] / stride_height

            """
            print(gws)
            print(ghs)
            tensor([[4.4375],
                    [3.5938]])
            tensor([[7.7812],
                    [9.8438]])
            """

            # 计算出正样本属于特征层的哪个特征点（由左下角的特征点预测）TODO: 这不是左上角吗
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)

            # 将真实框转换一个形式：num_true_box, 4
            gt_box = torch.FloatTensor(
                torch.cat(
                    [torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs]  # 前两个维度都为0，表示 iou 计算时，框的位置都在中心点
                    , 1)
            )

            """
            print(gt_box.shape) # torch.Size([2, 4])
            print(gt_box)
            tensor([[0.0000, 0.0000, 4.4375, 7.7812],
                   [0.0000, 0.0000, 3.5938, 9.8438]])
            """

            # 将先验框转换一个形式，9, 4
            anchor_shapes = torch.FloatTensor(
                torch.cat(
                    # 前两个维度都为0，表示 iou 计算时，框的位置都在中心点
                    (torch.zeros((self.num_anchors, 2)), torch.FloatTensor(scaled_anchors))
                    , 1)
            )

            """
            print(anchor_shapes.shape) # torch.Size([9, 4])
            print(anchor_shapes)
            tensor([[0.0000, 0.0000, 3.6250, 2.8125],
                    [0.0000, 0.0000, 4.8750, 6.1875],
                    [0.0000, 0.0000, 11.6562, 10.1875],
                    [0.0000, 0.0000, 0.9375, 1.9062],
                    [0.0000, 0.0000, 1.9375, 1.4062],
                    [0.0000, 0.0000, 1.8438, 3.7188],
                    [0.0000, 0.0000, 0.3125, 0.4062],
                    [0.0000, 0.0000, 0.5000, 0.9375],
                    [0.0000, 0.0000, 1.0312, 0.7188]])
            """

            #  计算交并比，num_true_box, 9，即真值框和每一个 anchor 框的 iou
            # （num_true_box, 4）*（9，4）->（num_true_box，9）
            anch_ious = jaccard(gt_box, anchor_shapes)

            """
            print(anch_ious.shape) # torch.Size([2, 9])
            print(anch_ious)
            tensor([[0.2953, 0.7374, 0.2908, 0.0518, 0.0789, 0.1986, 0.0037, 0.0136, 0.0215],
                    [0.2850, 0.5135, 0.2979, 0.0505, 0.0770, 0.1938, 0.0036, 0.0133, 0.0210]])
            """

            # 计算重合度最大的先验框是哪个，num_true_box, 1
            best_ns = torch.argmax(anch_ious, dim=-1)

            """
            print(best_ns.shape) # torch.Size([2])
            print(best_ns) # tensor([1, 1])
            """

            # 遍历所有物体，每个物体有一个最匹配的 anchor 框
            for j, best_n in enumerate(best_ns):  # i 遍历的是所有的图片，j 遍历的是一张图片的所有检测框
                if best_n not in anchor_index:  # 重合度最大的先验框不在当前特征层
                    continue

                # -------------------------------------------------------------#
                #   取出真实框各类坐标：
                #   gi和gj代表的是真实框对应的特征点的x轴y轴坐标
                #   gx和gy代表真实框的x轴和y轴坐标
                #   gw和gh代表真实框的宽和高
                # -------------------------------------------------------------#
                gi = gis[j].long()
                gj = gjs[j].long()
                gx = gxs[j]
                gy = gys[j]
                gw = gws[j]
                gh = ghs[j]

                # 真实框中点在特征层内
                if (gj < output_feature_height) and (gi < output_feature_width):  # 因为是左上角预测的
                    best_n = best_n - subtract_index  # 重合度最大的先验框的索引

                    # noobj_mask 代表无目标的特征点，置 0 表示有物体
                    noobj_mask[i, best_n, gj, gi] = 0
                    # mask 代表有目标的特征点，置 1 表示有物体
                    mask[i, best_n, gj, gi] = 1
                    # tx、ty 代表中心调整参数的真实值
                    tx[i, best_n, gj, gi] = gx - gi.float()  # 0~1
                    ty[i, best_n, gj, gi] = gy - gj.float()  # 0~1
                    # tw、th 代表宽高调整参数的真实值（TODO：这里进行了一些数学变换）
                    tw[i, best_n, gj, gi] = math.log(gw / scaled_anchors[best_n + subtract_index][0])
                    th[i, best_n, gj, gi] = math.log(gh / scaled_anchors[best_n + subtract_index][1])

                    # 用于获得 xywh 的比例，大目标loss权重小，小目标loss权重大
                    box_loss_scale_x[i, best_n, gj, gi] = target[i][j, 2]
                    box_loss_scale_y[i, best_n, gj, gi] = target[i][j, 3]

                    # tconf 代表物体置信度
                    tconf[i, best_n, gj, gi] = 1
                    # tcls 代表种类置信度
                    tcls[i, best_n, gj, gi, int(target[i][j, 4])] = 1
                else:
                    print('Step {0} out of bound'.format(i))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(
                        gj, output_feature_height, gi, output_feature_width))
                    continue

        return \
            mask, noobj_mask, \
            tx, ty, tw, th, \
            tconf, tcls, \
            box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction: torch.Tensor, target: torch.Tensor,
                   scaled_anchors: List[tuple], output_feature_height: int, output_feature_width: int,
                   noobj_mask: torch.Tensor):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        batch_size = len(target)

        # -------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        # -------------------------------------------------------#
        # [13, 26, 52] 中根据特征层宽获取特征层的编号，获得当前特征层先验框所属的编号，方便后面对先验框筛选
        anchor_index = [
            [0, 1, 2],  # 大特征层前三个
            [3, 4, 5],  # 中特征层
            [6, 7, 8]  # 小特征层
        ][self.feature_length.index(output_feature_width)]  # feature_length: [13, 26, 52]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]  # 获取当前特征层的先验框

        """
        print("scaled_anchors:\n", scaled_anchors)
        [[3.625    2.8125]
         [4.875    6.1875]
        [11.65625  10.1875]]
        """

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 处理是否使用 cuda
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, output_feature_width - 1, output_feature_width) \
            .repeat(output_feature_height, 1) \
            .repeat(int(batch_size * self.num_anchors / 3), 1, 1) \
            .view(x.shape) \
            .type(FloatTensor)

        grid_y = torch.linspace(0, output_feature_height - 1, output_feature_height) \
            .repeat(output_feature_width, 1) \
            .t() \
            .repeat(int(batch_size * self.num_anchors / 3), 1, 1) \
            .view(y.shape) \
            .type(FloatTensor)

        # print("grid_x.shape:",grid_x.shape) # grid_x.shape: torch.Size([16, 3, 13, 13])
        # print("grid_y.shape:",grid_y.shape)# grid_y.shape: torch.Size([16, 3, 13, 13])

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        """
        print(anchor_w, "\n",anchor_h)
        
        tensor([[ 3.6250],
        [ 4.8750],
        [11.6562]]) 
        tensor([[ 2.8125],
        [ 6.1875],
        [10.1875]])
        """

        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, output_feature_width * output_feature_height) \
            .view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, output_feature_width * output_feature_height) \
            .view(h.shape)

        # print("anchor_w.shape:",anchor_w.shape) # anchor_w.shape: torch.Size([16, 3, 13, 13])
        # print("anchor_h.shape:",anchor_h.shape) # anchor_h.shape: torch.Size([16, 3, 13, 13])

        # -------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        # TODO 宽高这里有数学变换
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        for i in range(batch_size):
            pred_boxes_for_ignore = pred_boxes[i]
            # -------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            # -------------------------------------------------------#

            # print("pred_boxes_for_ignore.shape:", pred_boxes_for_ignore.shape)

            # torch.Size([3, 13, 13, 4]) -> torch.Size([507, 4])
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)  # 预测框

            # print("pred_boxes_for_ignore.shape:", pred_boxes_for_ignore.shape)

            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            if len(target[i]) > 0:  # 真值框
                gx = target[i][:, 0:1] * output_feature_width
                gy = target[i][:, 1:2] * output_feature_height
                gw = target[i][:, 2:3] * output_feature_width
                gh = target[i][:, 3:4] * output_feature_height
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh], -1)).type(FloatTensor)

                # print("gt_box.shape:",gt_box.shape) # gt_box.shape: torch.Size([2, 4])

                # -------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                # -------------------------------------------------------#
                # torch.Size([target_size, 4]) * torch.Size([pred_size, 4]) -> torch.Size([target_size, pred_size])
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)

                # print("anch_ious.shape:",anch_ious.shape) # anch_ious.shape: torch.Size([2, 507])

                # -------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                # -------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious, dim=0)

                # print("anch_ious_max.shape:", anch_ious_max.shape)  # torch.Size([507])

                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])

                # print("anch_ious_max.shape:",anch_ious_max.shape) # torch.Size([3, 13, 13])

                noobj_mask[i][anch_ious_max > self.ignore_threshold] = 0  # 置 0 表示有物体

        return noobj_mask
