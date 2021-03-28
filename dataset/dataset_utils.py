from typing import List

import numpy
import PIL.Image

import torch
import torchvision


class Compose(object):
    """
    复合多种变换操作
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)

        return image, target


def get_transforms(config: dict, train: bool) -> Compose:
    """
    :param train: 是否是训练集，训练集包含额外的数据增强变换
    :return:
    """
    transforms = []
    transforms.append(ScaleImageAndTarget(config=config))
    transforms.append(NormImageAndTarget(config=config))
    if train:
        pass

    return Compose(transforms)


class ScaleImageAndTarget(object):
    """
    target 和 image 的 等比例放缩，target 归一化
    width * height * RGB -> channels(RGB) * width * height
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, raw_image: PIL.Image.Image, raw_target: numpy.ndarray) -> (PIL.Image.Image, numpy.ndarray):
        """
        :param raw_image: 原始图像
        :param raw_target: 原始标签
        :return: scaled_image, scaled_target
        """

        # 1. 图像原始大小，图像放缩后大小
        raw_width, raw_height = raw_image.size
        scaled_width = self.config["image_width"]
        scaled_height = self.config["image_height"]

        # 2. 计算图像放缩倍数，取最小的那个放缩值
        scale = min(scaled_width / raw_width, scaled_height / raw_height)

        # 3. 等比例放缩后的图像大小
        nw = int(raw_width * scale)
        nh = int(raw_height * scale)

        # 4. 图像等比例放缩
        scaled_image = raw_image.resize((nw, nh), PIL.Image.BICUBIC)

        # 5. 填补图像边缘
        new_image = PIL.Image.new('RGB', (scaled_width, scaled_height), (128, 128, 128))  # 创建一张灰色底板作为返回的图像
        new_image.paste(scaled_image, ((scaled_width - nw) // 2, (scaled_height - nh) // 2))  # 等比例放缩后的图像粘贴到底板中央

        # 6. 变换 target
        scaled_target = raw_target.copy()
        scaled_target[:, 0:4] = raw_target[:, 0:4] * scale
        scaled_target[:, 0] += (scaled_width - nw) // 2
        scaled_target[:, 1] += (scaled_height - nh) // 2

        return new_image, scaled_target


class NormImageAndTarget(object):
    """
    target 和 image 的 等比例放缩，target 归一化
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, scaled_image: PIL.Image.Image, scaled_target: numpy.ndarray) -> (
            numpy.ndarray, torch.FloatTensor):
        # 1. 归一化 PIL.Image.Image，width * height * RGB -> channels(RGB) * height * width
        norm_image = numpy.asarray(torchvision.transforms.ToTensor()(scaled_image))
        # 2. 归一化 target
        norm_target = scaled_target.copy()
        norm_target[:, 0] /= self.config["image_width"]
        norm_target[:, 1] /= self.config["image_height"]
        norm_target[:, 2] /= self.config["image_width"]
        norm_target[:, 3] /= self.config["image_height"]

        return norm_image, norm_target


def collate_fn(batch: List[tuple]) -> (torch.Tensor, torch.Tensor):
    """
    对一个批次的数据进行解包后打包
    数据集工具函数
    :param batch:
    :return:
    """
    # print("1:", type(batch), batch)                                 # batch 是一个返回值的数组：[(image, target), ……]
    # print("2:", *batch)                                             # *batch 将数组解包为：(image, target), ……
    # print("3:", type(zip(*batch)), list(zip(*batch)))               # zip 再次打包为：(image, ……) and (target, ……)
    norm_images, norm_targets = zip(*batch)

    tensord_images = torch.as_tensor(norm_images)
    tensord_target_list = [torch.as_tensor(norm_target) for norm_target in norm_targets]

    return tensord_images, tensord_target_list


def decode_tensord_target(config: dict, tensord_target: torch.Tensor) -> numpy.ndarray:
    """

    :param config:
    :param tensord_target:
    :return: box_num * (xmin, ymin, xmax, ymax, label)
    """
    raw_target = tensord_target.numpy()
    raw_target[:, 0] *= config["image_width"]
    raw_target[:, 1] *= config["image_height"]
    raw_target[:, 2] *= config["image_width"]
    raw_target[:, 3] *= config["image_height"]

    raw_boxes = raw_target.copy()
    raw_boxes[:, 0] = numpy.around(raw_target[:, 0] - raw_target[:, 2] / 2)
    raw_boxes[:, 1] = numpy.around(raw_target[:, 1] - raw_target[:, 3] / 2)
    raw_boxes[:, 2] = numpy.around(raw_target[:, 0] + raw_target[:, 2] / 2)
    raw_boxes[:, 3] = numpy.around(raw_target[:, 1] + raw_target[:, 3] / 2)

    return raw_boxes.astype(numpy.int)
