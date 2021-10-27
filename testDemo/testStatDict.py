# 在pytorch中，torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数，
# state_dict作为python的字典对象将每一层的参数映射成tensor张量，需要注意的是torch.nn.Module
# 模块中的state_dict只包含卷积层和全连接层的参数，当网络中存在batchnorm时，例如vgg网络结构，
# torch.nn.Module模块中的state_dict也会存放batchnorm's running_mean。
#
# torch.optim模块中的Optimizer优化器对象也存在一个state_dict对象，此处的state_dict字典对象
# 包含state和param_groups的字典对象，而param_groups key对应的value也是一个由学习率，动量等
# 参数组成的一个字典对象。因为state_dict本质上Python字典对象，所以可以很好地进行保存、更新、修改
# 和恢复操作（python字典结构的特性），从而为PyTorch模型和优化器增加了大量的模块化。

# 通过一个简单的案例来输出state_dict字典对象中存放的变量。

# encoding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F


# define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Initialize model
    model = TheModelClass()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # print model's state_dict
    print('Model.state_dict:')
    for param_tensor in model.state_dict():
        # 打印 key value字典
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    # print optimizer's state_dict
    print('Optimizer,s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])


if __name__ == '__main__':
    main()
