import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
from typing import Tuple

rnn = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
)
input = torch.randn(5, 3, 10)
# inputs = [torch.randn(1, 3) for _ in range(5)]
# print('inputs:', inputs)
print('input:', input.shape)
h0 = torch.randn(2, 3, 20)
print('h0:', h0.shape)
c0 = torch.randn(2, 3, 20)
print('c0:', c0.shape)
output, (hn, cn) = rnn(input, (h0, c0))
print('output:', output.shape)
print('hn:', hn.shape)
print('cn:', cn.shape)