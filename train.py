import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import yolov3net, yolov3loss
from conf import config
from util import train_utils
from dataset import dataset_utils, pennfudan_dataset

if __name__ == "__main__":
    Cuda = False  # 是否使用Cuda
    Normalize = True  # 是否对损失进行归一化

    Init_Epoch = 0  # 起始世代
    Freeze_Epoch = 50  # 冻结训练的世代
    Unfreeze_Epoch = 100  # 总训练世代

    # 提示 OOM 或者显存不足请调小 Batch_size
    Freeze_Epoch_Batch_Size = 16
    Unfreeze_Epoch_Batch_Size = 16

    Freeze_Epoch_LR = 1e-3
    Unfreeze_Epoch_LR = 1e-4

    Config = config.PennFudanConfig
    Config = config.DefaultConfig

    # 创建 yolo 模型，训练前一定要修改 Config 里面的 classes 参数
    # 训练的是 YoloNet 不是 Yolo
    model = yolov3net.YoloV3Net(Config)

    # 加载 darknet53 的权值作为预训练权值
    print('Loading weights into state dict...')

    model_path = "weights/demo_darknet53_weights.pth"
    model_path = "weights/demo_yolov3_weights.pth"
    device = torch.device('cuda' if Cuda and torch.cuda.is_available() else 'cpu')

    model_dict = model.state_dict()  # 模型权重
    pretrained_dict = torch.load(model_path, map_location=device)  # 预训练权重

    # 将 pretrained_dict 里不属于 model_dict 的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and np.shape(model_dict[k]) == np.shape(v)}

    model_dict.update(pretrained_dict)  # 更新现有的 model_dict
    model.load_state_dict(model_dict)  # 将更新有的模型权重加载回网络模型

    print('Loading weights into state dict Success！')

    # 开启训练模式
    net = model.train()

    # 使用 Cuda 则开启并行化
    if Cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            yolov3loss.YoloLoss(
                np.reshape(Config["anchors"], [-1, 2]),  # 转化为 anchor 列表，(3, 3, 2) -> (9, 2)
                Config["classes"],
                (Config["image_width"], Config["image_height"]),
                Cuda,
                Normalize
            )
        )

    if True:
        # 粗略训练预测头

        # 设置优化器
        optimizer = optim.Adam(net.parameters(), Freeze_Epoch_LR)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        pennFudanDataset = pennfudan_dataset.PennFudanDataset(
            '/Users/limengfan/Dataset/PennFudanPed',
            dataset_utils.get_train_transform(Config, False))

        train_data_loader = torch.utils.data.DataLoader(
            pennFudanDataset,
            batch_size=Freeze_Epoch_Batch_Size,
            shuffle=True,
            num_workers=5,
            collate_fn=dataset_utils.collate_fn)

        validate_data_loader = torch.utils.data.DataLoader(
            pennFudanDataset,
            batch_size=Freeze_Epoch_Batch_Size,
            shuffle=True,
            num_workers=5,
            collate_fn=dataset_utils.collate_fn)

        total_num = len(pennFudanDataset)
        num_train = len(pennFudanDataset)
        num_val = len(pennFudanDataset)
        batch_num = num_train // Freeze_Epoch_Batch_Size
        batch_num_val = num_val // Freeze_Epoch_Batch_Size

        # print("[Train] (num_train, Batch_size, batch_num):", num_train, Freeze_Epoch_Batch_Size, batch_num)

        # ------------------------------------#
        #   冻结特征网络
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            train_utils.fit_ont_epoch(
                net,  # 网络模型
                yolo_losses,  # 损失函数
                optimizer,  # 优化器
                epoch,  # 当前 epoch
                batch_num,  # 训练集批次数
                batch_num_val,  # 验证集批次数
                Freeze_Epoch,  # 总批次
                train_data_loader,  # 训练集
                validate_data_loader,  # 验证集
                Cuda  # 是否启用 Cuda
            )
            lr_scheduler.step()  # 更新步长

    if True:
        # 精细训练预测头和特征网络
        lr = 1e-4

        # 设置优化器
        optimizer = optim.Adam(net.parameters(), Unfreeze_Epoch_LR)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        pennFudanDataset = pennfudan_dataset.PennFudanDataset(
            '/Users/limengfan/Dataset/PennFudanPed',
            dataset_utils.get_train_transform(Config, False))

        train_data_loader = torch.utils.data.DataLoader(
            pennFudanDataset,
            batch_size=Unfreeze_Epoch_Batch_Size,
            shuffle=True,
            num_workers=5,
            collate_fn=dataset_utils.collate_fn)

        validate_data_loader = torch.utils.data.DataLoader(
            pennFudanDataset,
            batch_size=Unfreeze_Epoch_Batch_Size,
            shuffle=True,
            num_workers=5,
            collate_fn=dataset_utils.collate_fn)

        total_num = len(pennFudanDataset)
        num_train = len(pennFudanDataset)
        num_val = len(pennFudanDataset)
        batch_num = num_train // Unfreeze_Epoch_Batch_Size
        batch_num_val = num_val // Unfreeze_Epoch_Batch_Size

        # ------------------------------------#
        #   解冻特征网络
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            train_utils.fit_ont_epoch(
                net,  # 网络模型
                yolo_losses,  # 损失函数
                optimizer,  # 优化器
                epoch,  # 当前 epoch
                batch_num,  # 训练集批次数
                batch_num_val,  # 验证集批次数
                Unfreeze_Epoch,  # 总批次
                train_data_loader,  # 训练集
                validate_data_loader,  # 验证集
                Cuda  # 是否启用 Cuda
            )
            lr_scheduler.step()
