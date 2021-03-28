# -----------------------------------------------------------------------------------------------------------#
# PennFudanConfig
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
    "weights_path": '/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth',  # 模型权重
    # "weights_path": '/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/logs/'
    #                 'Epoch100-Total_Loss0.3207-Val_Loss0.3198.pth',  # 模型权重
    "confidence": 0.5,  # 正确预测框的最小置信度
    "iou": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": False,  # 是否使用 GPU
    "labels": [
        "person",
    ]
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
    "weights_path": '/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth',  # 模型权重
    "confidence": 0.5,  # 正确预测框的最小置信度
    "iou": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": False,  # 是否使用 GPU
    "labels": [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush"
    ]
}
