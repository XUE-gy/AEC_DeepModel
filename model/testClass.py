# pytorch lstm搭积木
import os
import torch
from torch import nn
# Define LSTM Neural Networks
# 同时对双通道进行回声消除，共用一套参数
class testClass(nn.Module):
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
        -numlayers：要堆栈的LSTM层
    """
    # 999,1024,999,1
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=2):
        super().__init__()
        self.myModel = nn.Sequential(
            # nn.Conv1d(in_channels=322, out_channels=322, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),

            # nn.LeakyReLU(0.2),
            # nn.Sigmoid()
        )

    def channelProcess(self, _x):
        return _x

    def forward(self, _x):
        (x0, x1) = _x.split(1, 1)
        x0 = x0.squeeze(dim=1)
        x1 = x1.squeeze(dim=1)
        _y = self.myModel(x0)
        print(_y)

        (x0, x1) = _x.split(1, 1)
        x0 = x0.squeeze(dim=1)
        x1 = x1.squeeze(dim=1)
        _y = self.myModel(x0)
        print(_y)


        return _y

def main2():
    random_seed = 123
    torch.manual_seed(random_seed)

    model = testClass(4030, 512, 4030, 1)

    x = torch.randn(4, 2, 2, 1)
    mulx = torch.randn(8, 2, 322, 4030)  # 输入 [8, 322, 999] # ([64, 4, 161, 4030])

    # print(x0.shape,x1.shape)

    y = model(x)  # 输出 [8, 161, 999]



if __name__ == "__main__":
    main2()

