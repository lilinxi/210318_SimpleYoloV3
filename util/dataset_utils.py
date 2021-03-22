import random
from typing import List

import numpy
from PIL import Image

import torch
from torchvision.transforms import functional

from util import yolo_utils
from model import config


def collate_fn(batch: List) -> tuple:
    """
    对一个批次的数据进行解包后打包
    数据集工具函数
    :param batch:
    :return:
    """
    # print("1:", type(batch), batch)                                 # batch 是一个返回值的数组：[(img, target), ……]
    # print("2:", *batch)                                             # *batch 将数组解包为：(img, target), ……
    # print("3:", type(zip(*batch)), list(zip(*batch)))               # zip 再次打包为：(img, ……) and (target, ……)
    # print("4:", type(tuple(zip(*batch))), tuple(zip(*batch)))       # tuple 返回其元组形式：(img, ……) and (target, ……)
    # exit(-1)
    """ 示例输出：
1: <class 'list'> [(<PIL.Image.Image image mode=RGB size=372x324 at 0x1279373D0>, {'labels': tensor([1])}), (<PIL.Image.Image image mode=RGB size=479x378 at 0x127C5BB20>, {'labels': tensor([1])})]
2: (<PIL.Image.Image image mode=RGB size=372x324 at 0x1279373D0>, {'labels': tensor([1])}) (<PIL.Image.Image image mode=RGB size=479x378 at 0x127C5BB20>, {'labels': tensor([1])})
3: <class 'zip'> <zip object at 0x10e232580>
4: <class 'tuple'> ((<PIL.Image.Image image mode=RGB size=372x324 at 0x1279373D0>, <PIL.Image.Image image mode=RGB size=479x378 at 0x127C5BB20>), ({'labels': tensor([1])}, {'labels': tensor([1])}))
    """

    return tuple(zip(*batch))


class ToTrainBGRNumpy(object):
    """
        Image( height * width )
    ->  Image( height * width )
    ->  Numpy( width * height * channel(RGB) )
    ->  Numpy( channel(RGB) * width * height )
    """

    def __call__(self, image: Image.Image, target: List) -> (numpy.ndarray, list):
        """
        :param image: <class 'PIL.Image.Image'>：height*width*RGB
        :param target: <class 'dict'>
        :return:
        """

        # 给图像增加灰条，实现不失真的resize
        crop_img = numpy.array(yolo_utils.letterbox_image(image, (config.Config["img_w"], config.Config["img_h"])))
        photo = numpy.transpose(crop_img, (2, 0, 1))

        return photo, target

        # Image.fromarray(crop_img, mode='RGB').show()
        # # 输入图像归一化到 0~1
        # photo = np.array(crop_img, dtype=np.float32) / 255.0
        #
        # # Image.open()读取的通道顺序是RGB
        # # 深度学习中普遍使用BGR而不用RGB
        # # RGB -> BGR
        # photo = np.transpose(photo, (2, 0, 1))


class ToTensor(object):
    """
    将 image 转化为 tensor 类型
    Image.open() 读取的通道顺序是 RGB，深度学习中普遍使用 BGR 而不用 RGB，之后需要进行了通道转换：RGB -> BGR
    width * height * channel(RGB) -> channel(RGB) * width * height
    """

    def __call__(self, image: Image.Image, target: List) -> (torch.Tensor, list):
        """
        :param image: <class 'PIL.Image.Image'>：height*width*RGB
        :param target: <class 'dict'>
        :return:
        """
        """
        print("[ToTensor] image.type", type(image))  # <class 'PIL.Image.Image'>
        print("[ToTensor] image.shape", numpy.array(image).shape)  # (536, 559, 3), width * height * channel(RGB)
        print("[ToTensor] image.size", image.size)  # (559, 536), height = 559, width = 536
        """

        """
        image.show()  # 显示图片
        Image.fromarray(numpy.array(image), mode='RGB').show()  # 显示图片
        """

        image = functional.to_tensor(
            image)  # Image( height * width ) -> Numpy( width * height * channel(RGB) ) -> Tensor( channel(RGB) * width * height )

        """
        Image.fromarray(
            (numpy.transpose(image.numpy(), (1, 2, 0)) * 255).astype(numpy.uint8),
            mode="RGB"
        ).show()  # 显示图片
        """

        """
        print("[ToTensor] image.type", type(image))  # <class 'torch.Tensor'>
        print("[ToTensor] image.shape", image.shape)  # torch.Size([3, 536, 559]), channel(RGB) * width * height
        """

        return image, target


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


class RandomHorizontalFlip(object):
    """
    随机水平反转
    """

    def __init__(self, prob):
        """
        :param prob: 反转的概率
        """
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: List) -> (torch.Tensor, list):
        if random.random() < self.prob:
            # TODO：待实现
            pass
            # image = image.flip(-1)  # 反转图像
            # bbox = target["boxes"]  # 反转包围盒
            # height, width = image.shape[-2:]
            # bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            # target["boxes"] = bbox
            # # 反转二值掩码
            # if "masks" in target:
            #     target["masks"] = target["masks"].flip(-1)
        return image, target


def get_transform(train: bool) -> Compose:
    """
    获取数据集的变换
    :param train: 是否是训练集
    :return:
    """
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


def get_train_transform(train: bool) -> Compose:
    """
    获取数据集的变换
    :param train: 是否是训练集
    :return:
    """
    transforms = []
    transforms.append(ToTrainBGRNumpy())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


###########################################################################
################################### Test ##################################
###########################################################################

if __name__ == "__main__":
    import torch.utils.data

    from util import dataset

    EPOCH = 2
    BATCH_SIZE = 1

    dataset = dataset.PennFudanDataset('/Users/limengfan/Dataset/PennFudanPed', get_train_transform(False))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 表示开启多少个线程数去加载你的数据，默认为0，代表只使用主进程
        collate_fn=collate_fn)

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (imgs, targets) in enumerate(data_loader):  # 分配 batch data, normalize x when iterate train_loader
            print("step(type(imgs), type(targets)):",
                  step, type(imgs), type(targets))  # <class 'tuple'> <class 'tuple'>
            exit(-1)
