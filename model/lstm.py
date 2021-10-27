# pytorch lstm搭积木
import os
import torch
from torch import nn
# Define LSTM Neural Networks
class LstmRNN(nn.Module):
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
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.myModel = nn.Sequential(
            # nn.Conv1d(in_channels=322, out_channels=322, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=322, out_channels=161, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2),
            # nn.Sigmoid()
        )
        self.forwardCalculation = nn.Linear(
            hidden_size,
            output_size,
        )
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        # self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        # 16,322,999 -> 16,322,999
        _y = self.myModel(_x)
        # 16,322,999 -> 16,322,1024
        x= _y
        # x, _ = self.lstm(_y)  # _x is input, size (seq_len, batch, input_size)
        # 16,322,1024 -> 16*322,1024
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # 16,322,1024 -> 16*322,1024
        x = x.reshape(s * b, h)
        # 16*322,1024 ->16,322,999
        x = self.forwardCalculation(x)
        # 16,322,999 -> 16,161,999
        x = x.view(s, b, -1)

        return x

def main2():
    model = LstmRNN(999, 1024, 999, 1)
    x = torch.randn(8, 322, 999)  # 输入 [8, 322, 999]
    y = model(x)  # 输出 [8, 161, 999]
    print(y.shape)


if __name__ == "__main__":
    main2()