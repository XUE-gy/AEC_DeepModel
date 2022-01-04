import torch
import torch.nn as nn

x = torch.randn(10, 16, 30, 32) # batch, channel , height , width
print(x.shape)
m = nn.Conv2d(16, 32, (3, 3), (1,1))  # in_channel, out_channel ,kennel_size,stride
print(m)
y = m(x)
print(y.shape)
