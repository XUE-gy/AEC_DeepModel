# pytorch lstm搭积木
import os
import torch
from torch import nn
# Define GRU Neural Networks

# 通道注意力机制
class ChannelAttention(nn.Module):
    # 参考 https://zhuanlan.zhihu.com/p/99261200?from=singlemessage，自己对二维通道做了改动，2d->1d
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 2d->1d
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 2d->1d

        self.fc1 = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)  # 2d->1d
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)  # 2d->1d

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)  # 2d->1d
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttGRU(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
        参数：
        -input_size：特征大小
        -hidden_size：隐藏单位的数量
        -output_size：输出的数量
        -numlayers：要堆栈的LSTN层
    """
    # 999,1024,999,1
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.myModel = nn.Sequential(
            # nn.Conv1d(in_channels=322, out_channels=322, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=322, out_channels=161, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # nn.Sigmoid()
        )
        self.inplanes = 161
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        # 全连接层
        self.forwardCalculation = nn.Linear(
            hidden_size,
            output_size,
        )
        self.Sigmoid = nn.Sequential(
            nn.Sigmoid()
        )
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        # self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        # 16,322,999 -> 16,322,999
        _y = self.myModel(_x)
        # print('y_size', list(_y.size()))
        # print('ca(_y).size', list(self.ca(_y).size()))
        _y = self.ca(_y) * _y
        # print('y_size', list(_y.size()))
        _y = self.sa(_y) * _y
        # print('y_size', list(_y.size()))
        # 16,322,999 -> 16,322,1024
        # x= _y
        x, _ = self.gru(_y)  # _x is input, size (seq_len, batch, input_size)
        # 16,322,1024 -> 16*322,1024
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # 16,322,1024 -> 16*322,1024
        x = x.reshape(s * b, h)
        # 16*322,1024 ->16,322,999
        x = self.forwardCalculation(x)
        x = self.Sigmoid(x)
        # 16,322,999 -> 16,161,999
        x = x.view(s, b, -1)

        return x

def main2():
    model = AttGRU(999, 999, 999, 1)
    x = torch.randn(8, 322, 999)  # 输入 [8, 322, 999]
    y = model(x)  # 输出 [8, 161, 999]
    print(y.shape)


if __name__ == "__main__":
    main2()