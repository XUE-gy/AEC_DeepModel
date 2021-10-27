# -*- coding:UTF-8 -*-
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import argparse
from tensorboardX import SummaryWriter

from data_preparation.data_preparation import FileDateset
from model.Baseline import Base_model
from model.lstm import LstmRNN
# from model.TCN_model import TCN_model

from model.ops import pytorch_LSD


def parse_ags1():
    parser = argparse.ArgumentParser()
    # 重头开始训练 defaule=None, 继续训练defaule设置为'/**.pth'
    parser.add_argument("--model_name", type=str, default=None, help="是否加载模型继续训练 '/50.pth' None")
    parser.add_argument("--batch-size", type=int, default=16, help="")
    parser.add_argument("--epochs", type=int, default=1000, help='20')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率 (default: 0.01,3e-4)')
    parser.add_argument('--train_data', default="./data_preparation/Synthetic/TRAIN", help='数据集的path')
    parser.add_argument('--val_data', default="./data_preparation/Synthetic/VAL4", help='验证样本的path')
    parser.add_argument('--checkpoints_dir', default="./checkpoints/AEC_baseline", help='模型检查点文件的路径(以继续培训)')
    parser.add_argument('--event_dir', default="./event_file/AEC_baseline", help='tensorboard事件文件的地址')
    args = parser.parse_args()
    return args


def main():
    # global train_loss # 可能有问题
    # global train_loss
    args = parse_ags1()
    # args = parser.parse_args(args=[])
    print("GPU是否可用：", torch.cuda.is_available())  # True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化 Dataset
    train_set = FileDateset(dataset_path=args.train_data)  # 实例化训练数据集
    val_set = FileDateset(dataset_path=args.val_data)  # 实例化验证数据集

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ###########    保存检查点的地址(如果检查点不存在，则创建)   ############
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    ################################
    #          实例化模型          #
    ################################
    # model = Base_model().to(device)  # 实例化模型
    model = Base_model().to(device) # 更换模型

    model = LstmRNN(999, 999, 999, 1).to(device)  # 更换模型

    # summary(model, input_size=(322, 999))  # 模型输出 torch.Size([64, 322, 999])
    # ###########    损失函数   ############
    # criterion = nn.MSELoss(reduce=True, size_average=True, reduction='mean')
    criterion = nn.MSELoss(reduce=True, size_average=True, reduction='mean')
    ###############################
    # 创建优化器 Create optimizers #
    ###############################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, )
    # lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

    # ###########    TensorBoard可视化 summary  ############
    # writer = SummaryWriter(args.event_dir)  # 创建事件文件

    # ###########    加载模型检查点   ############
    start_epoch = 0
    if args.model_name:
        print("加载模型：", args.checkpoints_dir + args.model_name)
        checkpoint = torch.load(args.checkpoints_dir + args.model_name)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint['epoch']
        # lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler

    for epoch in range(start_epoch, args.epochs):

        model.train()  # 训练模型
        # print(len(train_loader))
        # enumerate加入下标，batch_idx标识列表下标
        # train_X:远端语音和近端语音拼接 161+161=322
        # train_mask:近端语音和近端麦克风的平方和
        # train_nearend_mic_magnitude：近端麦克风
        # train_nearend_magnitude：近端语音
        epochLoss = 0
        # print('len(train_loader)', len(train_loader))
        for batch_idx, (train_X, train_mask, train_nearend_mic_magnitude, train_nearend_magnitude) in enumerate(
                train_loader):
            # 作用：若有cuda导入gpu
            train_X = train_X.to(device)  # 远端语音cat麦克风语音 [batch_size, 322, 999] (, F, T)
            train_mask = train_mask.to(device)  # IRM [batch_size 161, 999]
            train_nearend_mic_magnitude = train_nearend_mic_magnitude.to(device)
            train_nearend_magnitude = train_nearend_magnitude.to(device)

            # 前向传播
            pred_mask = model(train_X)  # [batch_size, 322, 999]--> [batch_size, 161, 999]

            # 调试信息
            # print('-------------------')
            # print('train_X.shape', train_X.shape)
            # print('train_mask.shape', train_mask.shape)
            # print('train_nearend_mic_magnitude.shape',train_nearend_mic_magnitude.shape)
            # print('train_nearend_magnitude.shape', train_nearend_magnitude.shape)
            # print('pred_mask.shape',pred_mask.shape)
            # 损失函数值
            # train_loss = criterion(pred_mask, train_mask)
            # train_loss = criterion(pred_mask, train_mask)

            # epochLoss = train_loss.item()

            # print('Loss:{:.5f}'.format(train_loss.item()))

            # 近端语音信号频谱 = mask * 麦克风信号频谱 [batch_size, 161, 999]
            pred_near_spectrum = pred_mask * train_nearend_mic_magnitude
            train_loss = criterion(pred_near_spectrum, train_nearend_magnitude)
            epochLoss = (epochLoss * batch_idx + train_loss.item()) / (batch_idx + 1)
            # train_lsd = pytorch_LSD(train_nearend_magnitude, pred_near_spectrum)
            # print('train_lsd', train_lsd)

            # 反向传播
            optimizer.zero_grad()  # 将梯度清零
            train_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # ###########    可视化打印   ############
        # print('Train Epoch: {} Loss: {:.6f} LSD: {:.6f}'.format(epoch + 1, train_loss.item(), train_lsd.item()))
        print('Train Epoch: {},Loss:{:.5f}'.format(epoch + 1,epochLoss))
        # print('Train Epoch: {}'.format(epoch + 1))
        # ###########    TensorBoard可视化 summary  ############
        # lr_schedule.step()  # 学习率衰减
        # writer.add_scalar(tag="lr", scalar_value=model.state_dict()['param_groups'][0]['lr'], global_step=epoch + 1)
        # writer.add_scalar(tag="train_loss", scalar_value=train_loss.item(), global_step=epoch + 1)
        # writer.add_scalar(tag="train_lsd", scalar_value=train_lsd.item(), global_step=epoch + 1)
        # writer.flush()

        # 神经网络在验证数据集上的表现
        model.eval()  # 测试模型
        # 测试的时候不需要梯度,并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）,强制之后的内容不进行计算图构建
        with torch.no_grad():
            epochLoss = 0
            for val_batch_idx, (val_X, val_mask, val_nearend_mic_magnitude, val_nearend_magnitude) in enumerate(
                    val_loader):
                val_X = val_X.to(device)  # 远端语音cat麦克风语音 [batch_size, 322, 999] (, F, T)
                val_mask = val_mask.to(device)  # IRM [batch_size 161, 999]
                val_nearend_mic_magnitude = val_nearend_mic_magnitude.to(device)
                val_nearend_magnitude = val_nearend_magnitude.to(device)


                # 前向传播
                val_pred_mask = model(val_X)
                # val_loss = criterion(val_pred_mask, val_mask)
                # val_loss = criterion(val_pred_mask, val_mask)


                # 近端语音信号频谱 = mask * 麦克风信号频谱 [batch_size, 161, 999]
                val_pred_near_spectrum = val_pred_mask * val_nearend_mic_magnitude
                # 计算真实近端语音和得到的近端语音标准差
                # val_lsd = pytorch_LSD(val_nearend_magnitude, val_pred_near_spectrum)
                val_loss = criterion(val_nearend_magnitude, val_pred_near_spectrum)
                epochLoss = (epochLoss * val_batch_idx + val_loss.item()) / (val_batch_idx + 1)
                # print('val_lsd',val_lsd)
            # ###########    可视化打印   ############
            # print('  val Epoch: {} \tLoss: {:.6f}\tlsd: {:.6f}'.format(epoch + 1, val_loss.item(), val_lsd.item()))
            # print('  val Epoch: {},val_loss:{:.5f} \t'.format(epoch + 1, val_loss.item()))
            print('  val Epoch: {},Loss:{:.5f} \t'.format(epoch + 1,epochLoss))
            ######################
            # 更新tensorboard    #
            ######################
            # writer.add_scalar(tag="val_loss", scalar_value=val_loss.item(), global_step=epoch + 1)
            # writer.add_scalar(tag="val_lsd", scalar_value=val_lsd.item(), global_step=epoch + 1)
            # writer.flush()

        # # ###########    保存模型   ############
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                # 'lr_schedule': lr_schedule.state_dict()
            }

            torch.save(checkpoint, '%s/%d_lstm.pth' % (args.checkpoints_dir, epoch + 1))



if __name__ == "__main__":
    main()


