import os

import numpy
from PIL import Image

import torch.utils.data

from util.dataset_utils import Compose


# -----------------------------------------------------------------------------------------------------------#
# 实现了行人数据集的封装
#
# Penn-Fudan 行人检测数据集：包含170个图像，其中包含345个行人实例
#
# class PennFudanDataset(torch.utils.data.Dataset):
#     def __init__(self, root: str, transforms: Compose) -> None:
#
# 实现了以下接口:
# 1. def __getitem__(self, idx: int) -> (numpy.ndarray, list):  # 根据索引获取数据集中的数据
# 2. def __len__(self) -> int:  # 获取数据集的长度
# -----------------------------------------------------------------------------------------------------------#


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms: Compose) -> None:
        """
        :param root: 数据集的根目录
        :param transforms: 数据增强需要的变换
        """
        self.root = root  # 数据集的根目录
        self.transforms = transforms  # 数据增强需要的变换
        # 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))  # 读取所有的图像的路径
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))  # 读取所有的蒙版的路径

    def __getitem__(self, idx: int) -> (numpy.ndarray, numpy.ndarray):  # 根据索引获取数据集中的数据
        """
        返回指定索引处的图片和标签
        :param idx:
        :return:
            - image: Numpy( channel(RGB) * width * height )
            - boxes: list([x, y, w, h, label])
        """
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])  # 加载图像索引下的路径
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])  # 加载蒙版索引下的路径

        image = Image.open(img_path).convert("RGB")  # 读取图像，转化为 RGB
        mask = Image.open(mask_path)  # 读取蒙版：0 为背景，非 0 为实例的掩码

        (image_height, image_width) = image.size

        mask = numpy.array(mask)  # 蒙版图像转化为 numpy 数组
        obj_ids = numpy.unique(mask)  # 获取所有的实例，每组相同的像素表示一个实例，（objects + 1, )
        obj_ids = obj_ids[1:]  # 去除 0，0 表示背景，并不表示实例，（objects, )
        masks = mask == obj_ids[:, None, None]  # 将 mask 转化为二值掩码的数组，每个实例一个二值掩码
        # （w, h）==（objects, 1, 1) -> （objects, w, h）

        # 对于每个二值掩码，获取其包围盒
        num_objs = len(obj_ids)  # 实例的数量
        boxes = []  # 包围盒数组，和二值掩码数组对齐
        for i in range(num_objs):
            pos = numpy.where(masks[i])  # 获取所有的掩码像素的坐标
            # 获取掩码的包围盒的坐标
            xmin = numpy.min(pos[1])
            xmax = numpy.max(pos[1])
            ymin = numpy.min(pos[0])
            ymax = numpy.max(pos[0])
            # 左上和右下坐标，转化为中心坐标和宽高，即 cx,cy,w,h
            cx = (xmax + xmin) / 2
            cy = (ymax + ymin) / 2
            w = xmax - xmin
            h = ymax - ymin
            boxes.append([cx, cy, w, h, 0])  # 这里并没有对实例进行分类，只有一类，所有的实例都为分类 1，其 index 为 0
        boxes = numpy.asarray(boxes)

        # 如果有数组增强变换，执行变换
        if self.transforms is not None:
            image, boxes = self.transforms(image, boxes)

        # 返回索引图像及其标签结果
        return image, boxes

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.imgs)


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    from util.dataset_utils import collate_fn

    EPOCH = 2
    BATCH_SIZE = 2

    dataset = PennFudanDataset('/Users/limengfan/Dataset/PennFudanPed', None)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 表示开启多少个线程数去加载你的数据，默认为0，代表只使用主进程
        collate_fn=collate_fn,
        drop_last=True)

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (images, targets) in enumerate(data_loader):  # 分配 batch data, normalize x when iterate train_loader
            print("step:", step)
            print(images)
            [print(target) for target in targets]
            exit(-1)
