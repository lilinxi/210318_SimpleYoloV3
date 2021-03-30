import tqdm

import torch
import torch.optim
import torch.utils.data.dataloader

import conf.config
import model.yolov3net, model.yolov3loss
import dataset.pennfudan_dataset


def train_one_epoch(
        yolov3_net: model.yolov3net.YoloV3Net,  # 网络模型
        yolov3_losses: model.yolov3loss.YoloLoss,  # 损失函数
        optimizer: torch.optim.Optimizer,  # 优化器
        epoch: int,  # 当前 epoch
        batch_num: int,  # 训练集的批次数，即为训练集大小除以批次大小
        batch_num_val: int,  # 验证集的批次数，即为验证集大小除以批次大小
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

    # step1.1, 打开网络训练模式
    yolov3_net = yolov3_net.train()

    # step1.3 加载 tadm 进度条，
    with tqdm.tqdm(total=batch_num, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict) as pbar:
        # step1.4 批次遍历数据集
        for iteration, (images, targets) in enumerate(train_data_loader):
            #     if iteration >= batch_num:  # 忽略最后一个不满 batch_size 的
            #         break
            #
            #     """
            #     print(type(images))  # <class 'tuple'>
            #     print(type(images[0]))  # <class 'numpy.ndarray'>
            #     print(images[0].shape)  # (3, 416, 416)
            #     """
            #
            #     """
            #     print(type(targets))  # <class 'tuple'>
            #     print(type(targets[0]))  # <class 'list'>
            #     print(type(targets[0][0]))  # <class 'list'>
            #     print(type(targets[0][0][0]))  # <class 'list'>
            #     for i, target in enumerate(targets): # 因为每张图片的 box 数目不同，所以各个标注的维度不同
            #         print(target)
            #     print(targets[0])  # [[159, 181, 301, 430, 1], [419, 170, 534, 485, 1]]
            #     """
            #
            #     # step1.5 图片 tuple 转化为 tensor Variable
            #     with torch.no_grad():  # 图片的转化过程中，没有梯度传递
            #         if cuda and torch.cuda.is_available():
            #             images = Variable(torch.as_tensor(images).type(torch.FloatTensor)).cuda()  # TODO：写到 collate_fn 里
            #             targets = [Variable(torch.as_tensor(ann).type(torch.FloatTensor)).cuda() for ann in targets]
            #         else:
            #             images = Variable(torch.as_tensor(images).type(torch.FloatTensor))
            #             targets = [Variable(torch.as_tensor(ann).type(torch.FloatTensor)) for ann in targets]
            #
            #     """
            #     print(type(images))  # <class 'torch.Tensor'>
            #     print(images.shape)  # torch.Size([16, 3, 416, 416])
            #     print(images.type())  # torch.FloatTensor
            #     """
            #
            #     """
            #     print(type(targets))  # <class 'list'>
            #     print(type(targets[0]))  # <class 'torch.Tensor'>
            #     """
            #
            #     # step1.6 清零梯度
            #     optimizer.zero_grad()
            #
            #     # step1.7 前向传播
            #     outputs = yolov3_net(images)
            #
            #     """
            #     print(type(outputs))  # <class 'tuple'>, (out0, out1, out2)  # 大，中，小
            #     print(type(outputs[0]))  # <class 'torch.Tensor'>
            #     print(outputs[0].shape)  # torch.Size([16, 255, 13, 13])
            #     print(outputs[1].shape)  # torch.Size([16, 255, 26, 26])
            #     print(outputs[2].shape)  # torch.Size([16, 255, 52, 52])
            #     """
            #
            #     # step1.8 计算损失
            #     losses = []
            #     num_pos_all = 0
            #
            #     for i in range(3):
            #         loss_item, num_pos = yolo_losses[i](outputs[i], targets)
            #         losses.append(loss_item)
            #         num_pos_all += num_pos
            #
            #     loss = sum(losses) / num_pos
            #
            #     # step1.9 反向传播
            #     loss.backward()  # 反向传播误差
            #     optimizer.step()  # 优化器进行优化
            #
            #     # TODO Note
            #     # optimizer.step()
            #     # 通常用在每个 mini-batch之中，而 scheduler.step() 通常用在 epoch 里面
            #     # 只有用了 optimizer.step()，模型才会更新，而 scheduler.step() 是对 lr 进行调整。
            #
            #     # step1.10 进度条更新
            #     train_loss += loss.item()  # 当前 epoch 的总损失
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

    '''
    # step2, 验证 #######################################################################################################
    
    yolov3_net.eval()
    with tqdm(total=batch_num_val, desc=f'Epoch(Validation) {epoch + 1}/{Epoch}', postfix=dict,
              mininterval=0.3) as pbar:
        for iteration, batch in enumerate(validate_data_loader):
            if iteration >= batch_num_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda and torch.cuda.is_available():
                    images_val = Variable(torch.as_tensor(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.as_tensor(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                else:
                    images_val = Variable(torch.as_tensor(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.as_tensor(ann).type(torch.FloatTensor)) for ann in targets_val]

                optimizer.zero_grad()
                outputs = yolov3_net(images_val)

                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos
                validate_loss += loss.item()

            pbar.set_postfix(**{'train_loss': validate_loss / (iteration + 1)})
            pbar.update(1)
    '''
    # -----------------------------------------------------------------------------------------------------------#
    # step3. 结果
    # -----------------------------------------------------------------------------------------------------------#

    # print('Finish Validation')
    # print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    # print('Total Loss: %.4f || Val Loss: %.4f ' % (train_loss / (batch_num + 1), validate_loss / (batch_num_val + 1)))
    # 
    # print('Saving state, iter:', str(epoch + 1))
    # 
    # torch.save(yolov3_net.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    #     (epoch + 1), train_loss / (batch_num + 1), validate_loss / (batch_num_val + 1)))


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

    # 2. 创建 yolo 模型，训练前一定要修改 Config 里面的 classes 参数，训练的是 YoloNet 不是 Yolo
    yolov3_net = model.yolov3net.YoloV3Net(Config)

    # 3. 加载 darknet53 的权值作为预训练权值
    load_pretrained_weights(yolov3_net, "weights/demo_darknet53_weights.pth", False)

    # 4. 开启训练模式
    yolov3_net = yolov3_net.train()

    # 5. 建立 loss 函数
    yolov3_loss = model.yolov3loss.YoloLoss(Config)

    # 6. 加载训练数据集和测试数据集
    train_data_loader = dataset.pennfudan_dataset.get_pennfudan_dataloader(
        config=Config,
        root="/Users/limengfan/Dataset/PennFudanPed",
        batch_size=Batch_Size,
        train=True,
        shuffle=True,
        num_workers=10,
    )
    train_batch_num = len(train_data_loader)

    validate_data_loader = dataset.pennfudan_dataset.get_pennfudan_dataloader(
        config=Config,
        root="/Users/limengfan/Dataset/PennFudanPed",
        batch_size=Batch_Size,
        train=True,
        shuffle=True,
        num_workers=10,
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
