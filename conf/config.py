import torch

# -----------------------------------------------------------------------------------------------------------#
# PennFudanConfig
# VocConfig
# DefaultConfig
# -----------------------------------------------------------------------------------------------------------#

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
    "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth",  # 模型权重
    "conf_threshold": 0.5,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": True and torch.cuda.is_available(),  # 是否使用 GPU
    "labels": [
        line.strip() for line in
        # open("/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/conf/voc.names").readlines()
        open("/home/lenovo/data/lmf/210318_SimpleYoloV3Sftp/conf/voc.names").readlines()
    ],
}

DefaultConfig: dict = {
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
        open("/home/lenovo/data/lmf/210318_SimpleYoloV3Sftp/conf/coco.names").readlines()
    ],
}
