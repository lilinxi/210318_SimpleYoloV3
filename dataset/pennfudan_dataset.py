import os

import numpy
import PIL.Image

import torch.utils.data

import dataset.dataset_utils


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
    def __init__(self, config: dict, root: str, train: bool = True) -> None:
        """
        :param config: YoloV3 的配置
        :param root: 数据集的根目录
        :param train: 是否是训练集，训练集包含额外的数据增强变换
        """
        super().__init__()

        # 1. 初始化数据集根目录和数据变换
        self.config = config
        self.root: str = root
        self.transforms: dataset.dataset_utils.Compose = dataset.dataset_utils.get_transforms(self.config, train)

        # 2. 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.images_dir = os.path.join(root, "PNGImages")
        self.masks_dir = os.path.join(root, "PedMasks")
        self.images_name = list(sorted(os.listdir(self.images_dir)))
        self.masks_name = list(sorted(os.listdir(self.masks_dir)))

    def __getitem__(self, idx: int) -> (PIL.Image.Image, numpy.ndarray):
        """
        返回指定索引处的图片和标签

        :param idx: 索引
        :return: (
                    scaled_image: PIL.Image.Image -> width * height * RGB,
                    scaled_target: numpy.ndarray -> true_box_num * (x, y, w, h, label)
                )
        """
        # 1. 拼接文件路径
        image_path = os.path.join(self.images_dir, self.images_name[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_name[idx])

        # 2. 读取图像文件和蒙版文件
        raw_image = PIL.Image.open(image_path).convert("RGB")  # 读取图像，转化为 RGB
        mask = PIL.Image.open(mask_path)  # 读取蒙版，为灰度图：0 为背景，非 0 为实例的掩码

        # 3. 解析蒙版文件，对每个蒙版，获取一个二值掩码，得到列表
        mask = numpy.array(mask)  # 蒙版图像转化为 numpy 数组，(w, h) -> (h, w)
        obj_ids = numpy.unique(mask)  # 获取所有的实例，每组相同的像素表示一个实例，(objects + 1, )
        obj_ids = obj_ids[1:]  # 去除 0，0 表示背景，并不表示实例，(objects, )
        # (h, w) == (objects, 1, 1) -> (objects, h, w)
        masks = mask == obj_ids[:, None, None]  # 将 mask 转化为二值掩码的数组，每个实例一个二值掩码

        # 4. 解析二值掩码，对于每个二值掩码，获取其包围盒，得到包围盒列表
        num_objs = len(obj_ids)  # 实例的数量
        raw_target = []  # 包围盒数组，和二值掩码数组对齐
        for i in range(num_objs):
            pos = numpy.where(masks[i])  # 获取所有的掩码像素的坐标，pos: (h, w)
            # 获取掩码的包围盒的坐标
            xmin = numpy.min(pos[1])
            xmax = numpy.max(pos[1])
            ymin = numpy.min(pos[0])
            ymax = numpy.max(pos[0])
            print("xmin,ymin,xmax,ymax:",xmin,ymin,xmax,ymax)
            # 左上和右下坐标，转化为中心坐标和宽高，即 raw_x, raw_y, raw_w, raw_h
            raw_x = (xmax + xmin) / 2
            raw_y = (ymax + ymin) / 2
            raw_w = xmax - xmin
            raw_h = ymax - ymin
            raw_target.append([raw_x, raw_y, raw_w, raw_h, 0])  # 这里并没有对实例进行分类，只有一类，所有的实例都为分类 1，其 index 为 0
        raw_target = numpy.asarray(raw_target)

        print("raw_target:",raw_target)

        # 5. 执行数据变换
        scaled_image, scaled_target = self.transforms(raw_image, raw_target)

        # 返回索引图像及其标签结果
        return scaled_image, scaled_target

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.images_name)


def get_pennfudan_dataloader(
        config: dict,
        root: str,
        batch_size: int,
        train: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    pennfudan_dataset = PennFudanDataset(
        config=config,
        root=root,
        train=train
    )

    pennfudan_dataloader = torch.utils.data.DataLoader(
        pennfudan_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.dataset_utils.collate_fn,
        drop_last=True
    )

    return pennfudan_dataloader


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    import conf.config

    EPOCH = 2
    BATCH_SIZE = 2

    pennfudan_dataloader = get_pennfudan_dataloader(
        config=conf.config.PennFudanConfig,
        root='/Users/limengfan/Dataset/PennFudanPed',
        batch_size=BATCH_SIZE
    )

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, tensord_target_list) in enumerate(pennfudan_dataloader):
            print("step:", step)
            print(tensord_images)
            print(tensord_target_list)
            exit(-1)
