创新点：
由于base论文的确是无懈可击，想办法在达到他的水平下（已实现），
一：减少他的网络模型提高效率，，保证结果尽量保持一致
二：对模型适当增删，保证效率不变的前提下，成为自己的模型：
    1.对loss函数进行修改
    2.对输入mask进行放缩进一步提高效率
    3.引入注意力机制替代模型部分lstm加快模型训练速度，还需要查看是否存在相似论文
    https://blog.csdn.net/weixin_52668444/article/details/115288690
三：重点说法，就是在去除模型部分结构效果更差但速度更快的基础上，加入注意力机制、gru，使得模型速度依旧快但是性能比删减后的模型强
    gru：改良的lstm，解决了lstm并行慢的问题，提高训练速度
    双向lstm（不理解，可以加快lstm的准确率但是不能提高速度）
    https://blog.csdn.net/xieyan0811/article/details/103491605


LOSS：
一，train loss与test loss结果分析：

train loss 不断下降，test loss不断下降，说明网络仍在学习;

train loss 不断下降，test loss趋于不变，说明网络过拟合;

train loss 趋于不变，test loss不断下降，说明数据集100%有问题;

train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;

train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。

gpu占用不满：
batch_size较小得原因，调大一点可以减少cpu搬运
