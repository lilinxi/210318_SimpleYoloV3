import numpy
from PIL import Image, ImageDraw

import torch


def display_image_box_mask(image: torch.Tensor, target: dict) -> None:
    """
    显示图片，包围盒和掩码
    :param image: torch.Tensor 格式的图片
    :param target:
    :return:
    """
    image = (numpy.transpose(image.numpy(), (1, 2, 0)) * 255).astype(
        numpy.uint8)  # channel(RGB) * width * height -> width * height * channel(RGB)

    image_copy = image  # 保存原始图像
    image = numpy.zeros_like(image_copy)  # 创建空白模板，在空白模板上叠加图像掩码

    for i, mask in enumerate(target["masks"]):  # 遍历所有的掩码
        image += image_copy * mask.numpy()[:, :, None]  # 点乘，叠加图像掩码

    image = Image.fromarray(image, mode='RGB')  # 将 image 转化为 Image 对象

    draw = ImageDraw.Draw(image)  # 创建 Image 对象的绘图对象
    for i, box in enumerate(target["boxes"]):  # 遍历所有的包围盒
        draw.rectangle(box.numpy(), None, 'red')  # 在 Image 上绘制包围盒

    image.show()  # 显示图片


def show_numpy_image(image: numpy.ndarray, show_channels=[]) -> None:
    """
    :param image: height * width * RGB or height * width * BGR
    :param show_channel: 判断是否 RGB，展示的通道数
    :return:
    """
    image = image.copy()
    for c in [0, 1, 2]:
        if c not in show_channels:
            image[:, :, c] = 0

    Image.fromarray(image, mode='RGB').show()
