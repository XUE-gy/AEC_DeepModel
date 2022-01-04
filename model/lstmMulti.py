# pytorch lstm搭积木
import os
import torch
from torch import nn
# Define LSTM Neural Networks
# 同时对双通道进行回声消除，共用一套参数


class LstmRNNMul(nn.Module):
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
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.myModel = nn.Sequential(
            # nn.Conv1d(in_channels=322, out_channels=322, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=322, out_channels=161, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),

            # nn.LeakyReLU(0.2),
            # nn.Sigmoid()
        )
        self.forwardCalculation = nn.Linear(
            hidden_size,
            output_size,
        )
        torch.manual_seed(1)
        self.cn = torch.randn(4, 2, 512).to(self.device)
        self.hn = torch.randn(4, 2, 512).to(self.device)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        # self.forwardCalculation = nn.Linear(hidden_size, output_size)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, _x):
        # 16,322,999 -> 16,322,999
        (x0, x1) = _x.split(1, 1)
        x0 = x0.squeeze(dim=1).cuda()
        x1 = x1.squeeze(dim=1).cuda()

        _y = self.myModel(x0)

        x, _ = self.lstm(_y)  # _x is input, size (seq_len, batch, input_size)
        # self.cn = cn
        # self.hn = hn
        # print('cn.shape',cn.shape)
        # print('hn.shape',hn.shape)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)

        x = x.reshape(s * b, h)

        x = self.forwardCalculation(x)

        x0 = x.view(s, b, -1)

        _y = self.myModel(x1)

        x, _ = self.lstm(_y)  # _x is input, size (seq_len, batch, input_size)

        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)

        x = x.reshape(s * b, h)

        x = self.forwardCalculation(x)

        x1 = x.view(s, b, -1)

        c = torch.cat([torch.unsqueeze(x0, dim=1), torch.unsqueeze(x1, dim=1)], dim=1)

        return c

def main2():
    model = LstmRNNMul(4030, 512, 4030, 1)



    x = torch.randn(8, 322, 4030)
    mulx = torch.randn(8, 2, 322, 4030)  # 输入 [8, 322, 999] # ([64, 4, 161, 4030])

    # print(x0.shape,x1.shape)

    y = model(mulx)  # 输出 [8, 161, 999]
    print(y.shape)


if __name__ == "__main__":
    main2()

    # _x
    # torch.Size([8, 322, 999])
    # _y
    # torch.Size([8, 161, 999])
    # x
    # torch.Size([8, 161, 999])
    # x.shape0
    # torch.Size([8, 161, 999])
    # x.shape1
    # torch.Size([1288, 999])
    # x.shape2
    # torch.Size([1288, 999])
    # x.shape3
    # torch.Size([8, 161, 999])
    # torch.Size([8, 161, 999])

    # print("_x", _x.shape)
    # _y = self.myModel(_x)
    # print("_y", _y.shape)
    # # 16,322,999 -> 16,322,1024
    # # x= _y
    # x, (hn, cn) = self.lstm(_y)  # _x is input, size (seq_len, batch, input_size)
    # print("x", x.shape)
    # print("hn", hn.shape)
    # print("cn", cn.shape)
    # # 16,322,1024 -> 16*322,1024
    # s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
    # print("x.shape0", x.shape)
    # # 16,322,1024 -> 16*322,1024
    # x = x.reshape(s * b, h)
    # print("x.shape1", x.shape)
    # # 16*322,1024 ->16,322,999
    # x = self.forwardCalculation(x)
    # print("x.shape2", x.shape)
    # # 16,322,999 -> 16,161,999
    # x = x.view(s, b, -1)
    # print("x.shape3", x.shape)