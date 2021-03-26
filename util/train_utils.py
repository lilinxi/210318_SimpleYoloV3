import numpy

import torch
from torch.autograd import Variable
from tqdm import tqdm

import torch.optim as optim

import torch.utils.data as data
from typing import List

from model import yolov3net, yolov3loss


def get_lr(optimizer):
    """
    获取优化器的 lr 参数
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net: yolov3net.YoloV3Net,
                  yolo_losses: List[yolov3loss.YoloLoss],
                  optimizer: optim.Optimizer,
                  epoch: int, batch_num: int, batch_num_val: int, Epoch: int,
                  train_data_loader: data.DataLoader, validate_data_loader: data.DataLoader,
                  cuda: bool):
    """
    训练一个 epoch
    :param net: 网络模型
    :param yolo_losses: 损失函数
    :param optimizer: 优化器
    :param epoch: 当前 epoch
    :param batch_num: 训练集的批次数，即为训练集大小除以批次大小
    :param batch_num_val: 验证集的批次数，即为验证集大小除以批次大小
    :param Epoch: 总批次
    :param train_data_loader: 训练集
    :param validate_data_loader: 验证集
    :param cuda: 是否启用 Cuda
    :return:
    """
    total_loss = 0  # # 当前 epoch 的总损失
    val_loss = 0  # 初始验证损失

    # step1, 训练 #######################################################################################################

    # step1.1, 打开网络训练模式
    net = net.train()
    # step1.2, 打开 cuda
    if cuda and torch.cuda.is_available():
        net = net.cuda()

    # step1.3 加载 tadm 进度条，
    with tqdm(total=batch_num, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        # step1.4 批次遍历数据集
        for iteration, (images, targets) in enumerate(train_data_loader):
            if iteration >= batch_num:  # 忽略最后一个不满 batch_size 的
                break

            """
            print(type(images))  # <class 'tuple'>
            print(type(images[0]))  # <class 'numpy.ndarray'>
            print(images[0].shape)  # (3, 416, 416)
            """

            """
            print(type(targets))  # <class 'tuple'>
            print(type(targets[0]))  # <class 'list'>
            print(type(targets[0][0]))  # <class 'list'>
            print(type(targets[0][0][0]))  # <class 'list'>
            for i, target in enumerate(targets): # 因为每张图片的 box 数目不同，所以各个标注的维度不同
                print(target)
            print(targets[0])  # [[159, 181, 301, 430, 1], [419, 170, 534, 485, 1]]
            """

            # step1.5 图片 tuple 转化为 tensor Variable
            with torch.no_grad():  # 图片的转化过程中，没有梯度传递
                if cuda and torch.cuda.is_available():
                    images = Variable(torch.as_tensor(images).type(torch.FloatTensor)).cuda() # TODO：写到 collate_fn 里
                    targets = [Variable(torch.as_tensor(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.as_tensor(images).type(torch.FloatTensor))
                    targets = [Variable(torch.as_tensor(ann).type(torch.FloatTensor)) for ann in targets]

            """
            print(type(images))  # <class 'torch.Tensor'>
            print(images.shape)  # torch.Size([16, 3, 416, 416])
            print(images.type())  # torch.FloatTensor
            """

            """
            print(type(targets))  # <class 'list'>
            print(type(targets[0]))  # <class 'torch.Tensor'>
            """

            # step1.6 清零梯度
            optimizer.zero_grad()

            # step1.7 前向传播
            outputs = net(images)

            """
            print(type(outputs))  # <class 'tuple'>, (out0, out1, out2)  # 大，中，小
            print(type(outputs[0]))  # <class 'torch.Tensor'>
            print(outputs[0].shape)  # torch.Size([16, 255, 13, 13])
            print(outputs[1].shape)  # torch.Size([16, 255, 26, 26])
            print(outputs[2].shape)  # torch.Size([16, 255, 52, 52])
            """

            # step1.8 计算损失
            losses = []
            num_pos_all = 0

            for i in range(3):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos

            # step1.9 反向传播
            loss.backward()  # 反向传播误差
            optimizer.step()  # 优化器进行优化

            # TODO Note
            # optimizer.step()
            # 通常用在每个 mini-batch之中，而 scheduler.step() 通常用在 epoch 里面
            # 只有用了 optimizer.step()，模型才会更新，而 scheduler.step() 是对 lr 进行调整。

            # step1.10 进度条更新
            total_loss += loss.item()  # 当前 epoch 的总损失
            pbar.set_postfix(
                **{
                    'total_loss': total_loss / (iteration + 1),  # 当前 epoch 的总损失 / 迭代次数
                    'lr': get_lr(optimizer)
                }
            )
            pbar.update(1)  # 进度条更新

    # torch.save(net.state_dict(), 'yolo_weights_epoch_' + str(epoch) + '.pkl')  # 只保存网络中的参数 (速度快, 占内存少)

    # step2, 验证 #######################################################################################################

    net.eval()
    with tqdm(total=batch_num_val, desc=f'Epoch(Validation) {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
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
                outputs = net(images_val)

                losses = []
                num_pos_all = 0
                for i in range(3):

                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (batch_num + 1), val_loss / (batch_num_val + 1)))

    print('Saving state, iter:', str(epoch + 1))

    torch.save(net.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (batch_num + 1), val_loss / (batch_num_val + 1)))
