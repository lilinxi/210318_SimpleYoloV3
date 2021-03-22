"""
实现了行人数据集的封装

Penn-Fudan 行人检测数据集：包含170个图像，其中包含345个行人实例

实现了以下接口:
1. def __getitem__(self, index)
2. def __len__(self)
"""

import os

import numpy
from PIL import Image

import torch
import torch.utils.data

from util.dataset_utils import Compose


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms: Compose):
        """
        :param root: 数据集的根目录
        :param transforms: 数据增强需要的变换
        """
        self.root = root  # 数据集的根目录
        self.transforms = transforms  # 数据增强需要的变换
        # 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))  # 读取所有的图像的路径
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))  # 读取所有的蒙版的路径

    def __getitem__(self, idx: int) -> (numpy.ndarray, list):  # 根据索引获取数据集中的数据
        """
        返回指定索引处的图片和标签
        :param idx:
        :return:
            - image: Numpy( channel(RGB) * width * height )
            - boxes: list([xmin, ymin, xmax, ymax, cls])
        """
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])  # 加载图像索引下的路径
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])  # 加载蒙版索引下的路径

        image = Image.open(img_path).convert("RGB")  # 读取图像，转化为 RGB
        mask = Image.open(mask_path)  # 读取蒙版：0 为背景，非 0 为实例的掩码

        mask = numpy.array(mask)  # 蒙版图像转化为 numpy 数组
        obj_ids = numpy.unique(mask)  # 获取所有的实例，每组相同的像素表示一个实例
        obj_ids = obj_ids[1:]  # 去除 0，0 表示背景，并不表示实例
        masks = mask == obj_ids[:, None, None]  # 将 mask 转化为二值掩码的数组，每个实例一个二值掩码

        # TODO：Note
        # print(obj_ids.shape, obj_ids[:, None, None].shape, mask.shape, masks.shape) # (2,) (2, 1, 1) (341, 414) (2, 341, 414)
        # print(obj_ids, obj_ids[:, None, None])
        # exit(-1)

        """ 旧的目标标识
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
            # 添加掩码的包围盒，和二维掩码数组对齐
            boxes.append([xmin, ymin, xmax, ymax])

        # 将掩码数组，包围盒数组，标签数组都转化为 torch.Tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 这里并没有对实例进行分类，只有一类，所有的实例都为分类 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 图像的标签结果
        target = {}
        target["masks"] = masks
        target["labels"] = labels
        target["boxes"] = boxes
        """

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
            cx = (xmax - xmin) / 2
            cy = (ymax - ymin) / 2
            w = xmax - xmin
            h = ymax - ymin
            boxes.append([cx, cy, w, h, 0])  # 这里并没有对实例进行分类，只有一类，所有的实例都为分类 1，其 index 为 0

        # 如果有数组增强变换，执行变换
        if self.transforms is not None:
            image, boxes = self.transforms(image, boxes)

        # 返回索引图像及其标签结果
        return image, boxes

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.imgs)


###########################################################################
################################### Test ##################################
###########################################################################

if __name__ == "__main__":
    from util.dataset_utils import collate_fn, get_transform
    from util.display_utils import display_image_box_mask

    # TODO Note: 必须在主函数里，这样，之后才能 fork 多线程
    # 为使用了 multiprocessing  的程序，提供冻结以产生 Windows 可执行文件的支持。
    # 需要在 main 模块的 if __name__ == '__main__' 该行之后马上调用该函数。
    # 由于Python的内存操作并不是线程安全的，对于多线程的操作加了一把锁。这把锁被称为GIL（Global Interpreter Lock）。
    # 而 Python 使用多进程来替代多线程
    # torch.multiprocessing.freeze_support()
    #
    # torch.manual_seed(1)  # fake random makes reproducible

    EPOCH = 2
    BATCH_SIZE = 1

    dataset = PennFudanDataset('/Users/limengfan/Dataset/PennFudanumpyed', get_transform(False))

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
            print("step(len(imgs)):", step, len(imgs))
            print("step(len(targets)):", step, len(targets))
            print("step(type(imgs[0])):", step, type(imgs[0]))  # <class 'torch.Tensor'>
            print("step(imgs[0].shape):", step, imgs[0].shape)  # torch.Size([3, 536, 559])
            display_image_box_mask(imgs[0], targets[0])
            exit(-1)
