
import numpy as np
import torch
import torch.optim as optim

if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 3], [4, 4]])

    # loss(xi,yi) = (xi-yi)^2
    # 1,1.,1.,0;
    #reduce = true,则看size_average，若true取平均，false取sum
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)

    input = torch.autograd.Variable(torch.from_numpy(a))
    target = torch.autograd.Variable(torch.from_numpy(b))

    loss = loss_fn(input.float(), target.float())

    print(loss)




    # loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    # # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    # # loss_fn = torch.nn.MSELoss()
    # input = torch.autograd.Variable(torch.randn(3, 4))
    # target = torch.autograd.Variable(torch.randn(3, 4))
    # loss = loss_fn(input, target)
    # print(input);
    # print(target);
    # print(loss)
    # print(input.size(), target.size(), loss.size())




