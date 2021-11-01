# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 上午9:54
@file: finetune.py
@author: zj
@description: 
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from utils.data.custom_finetune_dataset import CustomFinetuneDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir


def load_data(data_root_dir):
    transform = transforms.Compose([    # 串联多个图片变换的操作
        transforms.ToPILImage(),   # Tensor转化成PILImage类型进行图片处理
        transforms.Resize((227, 227)),  # 功能：重置图像分辨率
        transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
        transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 功能：对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
    ])
# ndarray的shape H x W x C，Tensor 的shape为 C x H x W，ToPILImage将HWC变成了WH，去掉了通道这个维度
    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name) # data_dir:'./data/finetune_car\\train'
        data_set = CustomFinetuneDataset(data_dir, transform=transform)
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)    #
        # 每次批量处理，其中batch_positive个正样本，batch_negative个负样本
        # data_set.get_positive_num()：@param num_positive: 正样本数目   66122
        # data_set.get_negative_num()：@param num_negative: 负样本数目   454839
        #  32：@param batch_positive: 单批次正样本数  32
        #  96：@param batch_negative: 单批次负样本数  96
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True) #
        # sampler：样本抽样  num_workers=0：使用多进程加载的进程数，0代表不适用多进程
        # drop_last=False：dataset中的数据个数可能不是batch_size的整数倍，drop_last为true会将多出来不足一个batch的数据丢弃
        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()    # 数量

    return data_loaders, data_sizes


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=5, device=None):
    since = time.time()   # 计算时间（执行速度）

    best_model_weights = copy.deepcopy(model.state_dict())
    # copy.deepcopy()  深复制  什么都是复制过来 创建独立内存空间。不跟着原数据的改变而改变。
    # torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数
    best_acc = 0.0

    for epoch in range(num_epochs):   # 25
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode  训练集用
            else:
                model.eval()  # Set model to evaluate mode  验证集和测试集用
# model.train()和model.eval()的区别主要在于Batch Normalization和Dropout两层。
# 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。
# model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
            """如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。
            model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
            对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
"""
            running_loss = 0.0    # 损失统计
            running_corrects = 0  # 正确率计数

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:   # 在训练或验证时输入图片和标签
                inputs = inputs.to(device)     # 调用GPU
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # 三个函数的作用是先将梯度归零（optimizer.zero_grad()），
                # 然后反向传播计算得到每个参数的梯度值（loss.backward()），
                # 最后通过梯度下降执行一步参数更新（optimizer.step()）

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # 影响网络的自动求导机制，也就是网络前向传播后不会进行求导和进行反向传播。另外他不会影响dropout层和batchnorm层。
                    outputs = model(inputs)  # 前向传播计算预测值
                    _, preds = torch.max(outputs, 1)#
                    # 不加下划线表示返回一行中最大的数，加下划线表示返回一行中最大数的索引
                    loss = criterion(outputs, labels)  # 计算损失

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)  # 128
                # item()方法是得到一个元素张量里面的元素值
                # 具体就是用于将一个零维张量转换成浮点数，比如计算loss，accuracy的值
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()
            # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，
            # 可以根据具体的需求来做，只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。
            epoch_loss = running_loss / data_sizes[phase]   # data_sizes: {'train': 520960, 'val': 471552}
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断GPU是否可用    device:cpu

    data_loaders, data_sizes = load_data('./data/finetune_car')
    #  data_loaders:(train,val)   data_sizes:{'train': 520960}
    model = models.alexnet(pretrained=True) # 调用Alexnet网络
    # print(model)
    num_features = model.classifier[6].in_features  # num_features=4096
    model.classifier[6] = nn.Linear(num_features, 2)   # 把Linear的out_features  1000改为2
    # print(model)
    model = model.to(device) # 加载网络模型

    criterion = nn.CrossEntropyLoss()  # 计算交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)   # 优化器  优化batch_size
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 有序调整学习率   优化epoch

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=5)   # 训练
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')
