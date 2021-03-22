Config: dict = {
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
    "cuda": False  # 是否使用 GPU
}
