import os

import torch

# -----------------------------------------------------------------------------------------------------------#
# PennFudanConfig
# VocConfig
# DefaultCocoConfig
# -----------------------------------------------------------------------------------------------------------#

CocoNamesPath = os.path.join(os.getcwd(), "conf", "coco.names")
VocNamesPath = os.path.join(os.getcwd(), "conf", "voc.names")

DarkNet53WeightPath = os.path.join(os.getcwd(), "weights", "demo_darknet53_weights.pth")

TrainLogPath = os.path.join(os.getcwd(), "logs_voc")

VocDatasetRoot = "/Users/limengfan/Dataset/VOC/VOC2012Train"
# VocDatasetRoot = "/home/lenovo/data/lmf/Dataset/voc/VOCtrainval_11-May-2012"

PennFudanConfig: dict = {
    "anchors": [  # 锚框，width * height
        [
            [116, 90], [156, 198], [373, 326]  # 大
        ], [
            [30, 61], [62, 45], [59, 119]  # 中
        ], [
            [10, 13], [16, 30], [33, 23]  # 小
        ]
    ],
    "classes": 1,  # 分类数目
    "image_height": 416,  # 输入图片高度
    "image_width": 416,  # 输入图片宽度
    # "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth",  # 模型权重
    "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/logs/"
                    "Epoch88-Train_Loss5.0447-Val_Loss2.9787.pth",  # 模型权重
    "conf_threshold": 0.5,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": False and torch.cuda.is_available(),  # 是否使用 GPU
    "labels": [
        "person",
    ]
}

VocConfig: dict = {
    "anchors": [  # 锚框，width * height
        [
            [116, 90], [156, 198], [373, 326]  # 大
        ], [
            [30, 61], [62, 45], [59, 119]  # 中
        ], [
            [10, 13], [16, 30], [33, 23]  # 小
        ]
    ],
    "classes": 20,  # 分类数目
    "image_height": 416,  # 输入图片高度
    "image_width": 416,  # 输入图片宽度
    # "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth",  # 模型权重
    "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/logs_voc/"
    # "weights_path": "/home/lenovo/data/lmf/210318_SimpleYoloV3Sftp/logs_voc/"
                    "Epoch223-Train_Loss0.0166-Val_Loss19.6223.pth",  # 模型权重
    "conf_threshold": 0.05,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": True and torch.cuda.is_available(),  # 是否使用 GPU
    "labels": [
        line.strip() for line in
        # open("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/conf/voc.names").readlines()
        # open("/home/lenovo/data/lmf/210318_SimpleYoloV3Sftp/conf/voc.names").readlines()
        open(VocNamesPath).readlines()
    ] if os.path.exists(VocNamesPath)
    else [print("warn in loading voc.names", VocNamesPath)],
}

DefaultCocoConfig: dict = {
    "anchors": [  # 锚框，width * height
        [
            [116, 90], [156, 198], [373, 326]  # 大
        ], [
            [30, 61], [62, 45], [59, 119]  # 中
        ], [
            [10, 13], [16, 30], [33, 23]  # 小
        ]
    ],
    "classes": 80,  # 分类数目
    "image_height": 416,  # 输入图片高度
    "image_width": 416,  # 输入图片宽度
    "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth",  # 模型权重
    "conf_threshold": 0.5,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": False and torch.cuda.is_available(),  # 是否使用 GPU
    "labels": [
        line.strip() for line in
        # open("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/conf/coco.names").readlines()
        # open("/home/lenovo/data/lmf/210318_SimpleYoloV3Sftp/conf/coco.names").readlines()
        open(CocoNamesPath).readlines()
    ] if os.path.exists(CocoNamesPath)
    else [print("warn in loading coco.names", CocoNamesPath)],
}
