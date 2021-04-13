import numpy
import PIL.Image

import torch.utils.data
import torchvision

import conf.config
import dataset.dataset_utils


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict, root: str, image_set: str, train: bool = True) -> None:
        super().__init__()

        self.config = config
        self.voc2012_dataset = torchvision.datasets.VOCDetection(
            root=root,
            image_set=image_set,
        )
        self.transforms: dataset.dataset_utils.Compose = dataset.dataset_utils.get_transforms(self.config, train)

    def __getitem__(self, idx: int) -> (PIL.Image.Image, numpy.ndarray):
        (raw_image, raw_annotation) = self.voc2012_dataset[idx]
        raw_target = []
        for object in raw_annotation["annotation"]["object"]:
            xmin, ymin, xmax, ymax = \
                int(object["bndbox"]["xmin"]), \
                int(object["bndbox"]["ymin"]), \
                int(object["bndbox"]["xmax"]), \
                int(object["bndbox"]["ymax"])
            raw_x = (xmax + xmin) / 2
            raw_y = (ymax + ymin) / 2
            raw_w = xmax - xmin
            raw_h = ymax - ymin
            raw_target.append(
                [
                    raw_x, raw_y, raw_w, raw_h,
                    self.config["labels"].index(object["name"]) if object["name"] in self.config["labels"] else -1
                ]
            )
        raw_target = numpy.asarray(raw_target)

        # 5. 执行数据变换
        scaled_image, scaled_target = self.transforms(raw_image, raw_target)

        # 返回索引图像及其标签结果
        return scaled_image, scaled_target

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.voc2012_dataset)


def get_voc_dataloader(
        config: dict,
        image_set: str,
        batch_size: int = 1,
        train: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    voc_dataset = VOCDataset(
        config=config,
        root=conf.config.VocDatasetRoot,
        image_set=image_set,
        train=train
    )

    voc_dataloader = torch.utils.data.DataLoader(
        voc_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.dataset_utils.collate_fn,
        drop_last=True
    )

    return voc_dataloader


def get_voc_train_dataloader(
        config: dict,
        batch_size: int = 1,
        train: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return get_voc_dataloader(
        config,
        "train",
        batch_size,
        train,
        shuffle,
        num_workers,
    )


def get_voc_eval_dataloader(
        config: dict,
        batch_size: int = 1,
        train: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return get_voc_dataloader(
        config,
        "val",
        batch_size,
        train,
        shuffle,
        num_workers,
    )


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    import conf.config

    EPOCH = 2

    voc_dataloader = get_voc_train_dataloader(
        config=conf.config.VocConfig,
    )

    print(len(voc_dataloader))
    print(len(get_voc_eval_dataloader(config=conf.config.VocConfig)))

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, tensord_target_list) in enumerate(voc_dataloader):
            print("step:", step)
            print(tensord_images)
            print(tensord_target_list)
            exit(-1)
