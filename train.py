import tqdm

import torch
import torch.optim
import torch.utils.data.dataloader

import conf.config
import model.yolov3net, model.yolov3loss
import dataset.pennfudan_dataset


def train_one_epoch(
        yolov3_net: model.yolov3net.YoloV3Net,  # 网络模型
        yolov3_losses: model.yolov3loss.YoloV3Loss,  # 损失函数
        optimizer: torch.optim.Optimizer,  # 优化器
        epoch: int,  # 当前 epoch
        train_batch_num: int,  # 训练集的批次数，即为训练集大小除以批次大小
        validate_batch_num: int,  # 验证集的批次数，即为验证集大小除以批次大小
        total_epoch: int,  # 总批次
        train_data_loader: torch.utils.data.dataloader.DataLoader,  # 训练集
        validate_data_loader: torch.utils.data.dataloader.DataLoader,  # 验证集
) -> None:
    """
    训练一个 epoch
    :return:
    """

    # -----------------------------------------------------------------------------------------------------------#
    # step1. 训练
    # -----------------------------------------------------------------------------------------------------------#
    total_train_loss = 0  # 当前 epoch 的训练总损失

    # 1. 打开网络训练模式
    yolov3_net = yolov3_net.train()

    # 2. 加载 tadm 进度条，
    with tqdm.tqdm(total=train_batch_num, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict) as pbar:
        # 3. 批次遍历数据集
        for iteration, (tensord_images, tensord_target_list) in enumerate(train_data_loader):
            # 4. 清零梯度
            optimizer.zero_grad()

            # 5. 前向传播
            predict_feature_list = yolov3_net(tensord_images)

            # 6. 计算损失
            loss = yolov3_losses(predict_feature_list, tensord_target_list)

            # 7. 反向传播
            loss.backward()

            # 8. 优化器优化参数
            optimizer.step()

            # 9. 进度条更新
            total_train_loss += loss.item()
            pbar.set_postfix(
                **{
                    'lr': optimizer.param_groups[0]["lr"],  # 优化器的当前学习率
                    'train_loss': total_train_loss / (iteration + 1),  # 当前 epoch 的训练总损失 / 迭代次数
                }
            )
            pbar.update(1)  # 进度条更新

    # -----------------------------------------------------------------------------------------------------------#
    # step2. 验证
    # -----------------------------------------------------------------------------------------------------------#
    total_validate_loss = 0  # 当前 epoch 的验证总损失

    # 1. 打开网络验证模式
    yolov3_net = yolov3_net.eval()

    # 2. 加载 tadm 进度条，
    with tqdm.tqdm(total=validate_batch_num, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict) as pbar:
        # 3. 批次遍历数据集
        for iteration, (tensord_images, tensord_target_list) in enumerate(validate_data_loader):
            # 4. 清零梯度
            optimizer.zero_grad()

            # 5. 前向传播
            predict_feature_list = yolov3_net(tensord_images)

            # 6. 计算损失
            loss = yolov3_losses(predict_feature_list, tensord_target_list)

            # 7. 进度条更新
            total_validate_loss += loss.item()
            pbar.set_postfix(
                **{
                    'validate_loss': total_validate_loss / (iteration + 1),  # 当前 epoch 的验证总损失 / 迭代次数
                }
            )
            pbar.update(1)  # 进度条更新

    # -----------------------------------------------------------------------------------------------------------#
    # step3. 结果
    # -----------------------------------------------------------------------------------------------------------#
    # 1. 计算平均损失
    train_loss = total_train_loss / train_batch_num
    validate_loss = total_validate_loss / validate_batch_num

    # 2. 显示结果
    ret = "Epoch%d-Train_Loss%.4f-Val_Loss%.4f" % (epoch + 1, train_loss, validate_loss)
    print(ret)

    # 3. 保存权重
    torch.save(yolov3_net.state_dict(), "logs/" + ret + ".pth")


def load_pretrained_weights(net: torch.nn.Module, weights_path: str, cuda: bool):
    """
    加载预训练权重中名称相符的部分

    :param net: 网络
    :param weights_path: 预训练权重路径
    :param cuda: 是否使用 gpu
    :return:
    """
    print('Loading weights into state dict...')

    # 1. 确定设备
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    # 2. 获取网络权重字典和预训练权重字典
    net_dict = net.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=device)

    # 3. 将 pretrained_dict 里不属于 net_dict 的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}

    # 4. 将 pretrained_dict 的键值更新到 net_dict
    net_dict.update(pretrained_dict)

    # 5. net 加载 net_dict
    net.load_state_dict(net_dict)

    print('Loading weights into state dict Success！')


if __name__ == "__main__":
    # 0. 确保每次的伪随机数相同以便于问题的复现
    torch.manual_seed(1)

    # 1. 训练参数
    Config = conf.config.PennFudanConfig

    # 提示 OOM 或者显存不足请调小 Batch_size
    Batch_Size = 16

    Init_Epoch = 0  # 起始世代
    Freeze_Epoch = 50  # 冻结训练的世代
    Unfreeze_Epoch = 100  # 总训练世代

    Freeze_Epoch_LR = 1e-3
    Unfreeze_Epoch_LR = 1e-4

    Num_Workers = 10
    Suffle = True

    # 2. 创建 yolo 模型，训练前一定要修改 Config 里面的 classes 参数，训练的是 YoloNet 不是 Yolo
    yolov3_net = model.yolov3net.YoloV3Net(Config)

    # 3. 加载 darknet53 的权值作为预训练权值
    load_pretrained_weights(yolov3_net, "weights/demo_darknet53_weights.pth", False)

    # 4. 开启训练模式
    yolov3_net = yolov3_net.train()

    # 5. 建立 loss 函数
    yolov3_loss = model.yolov3loss.YoloV3Loss(Config)

    # 6. 加载训练数据集和测试数据集
    train_data_loader = dataset.pennfudan_dataset.get_pennfudan_dataloader(
        config=Config,
        root="/Users/limengfan/Dataset/PennFudanPed",
        batch_size=Batch_Size,
        train=True,
        shuffle=Suffle,
        num_workers=Num_Workers,
    )
    train_batch_num = len(train_data_loader)

    validate_data_loader = dataset.pennfudan_dataset.get_pennfudan_dataloader(
        config=Config,
        root="/Users/limengfan/Dataset/PennFudanPed",
        batch_size=Batch_Size,
        train=True,
        shuffle=Suffle,
        num_workers=Num_Workers,
    )
    validate_batch_num = len(validate_data_loader)

    # 7. 粗略训练预测头

    # 7.1 优化器和学习率调整器
    optimizer = torch.optim.Adam(yolov3_net.parameters(), Freeze_Epoch_LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 7.2 冻结特征网络
    for param in yolov3_net.backbone.parameters():
        param.requires_grad = False

    # 7.3 训练若干 Epoch
    for epoch in range(Init_Epoch, Freeze_Epoch):
        train_one_epoch(
            yolov3_net,  # 网络模型
            yolov3_loss,  # 损失函数
            optimizer,  # 优化器
            epoch,  # 当前 epoch
            train_batch_num,  # 训练集批次数
            validate_batch_num,  # 验证集批次数
            Freeze_Epoch,  # 总批次
            train_data_loader,  # 训练集
            validate_data_loader,  # 验证集
        )
        lr_scheduler.step()  # 更新步长

    # 8. 精细训练预测头和特征网络

    # 8.1 优化器和学习率调整器
    optimizer = torch.optim.Adam(yolov3_net.parameters(), Unfreeze_Epoch_LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 8.2 解冻特征网络
    for param in yolov3_net.backbone.parameters():
        param.requires_grad = True

    # 8.3 训练若干 Epoch
    for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
        train_one_epoch(
            yolov3_net,  # 网络模型
            yolov3_loss,  # 损失函数
            optimizer,  # 优化器
            epoch,  # 当前 epoch
            train_batch_num,  # 训练集批次数
            validate_batch_num,  # 验证集批次数
            Unfreeze_Epoch,  # 总批次
            train_data_loader,  # 训练集
            validate_data_loader,  # 验证集
        )
        lr_scheduler.step()  # 更新步长
