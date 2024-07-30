import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
import utils

batch_size = 512

# step 1、 load dataset
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data',
                                                                      train=True,
                                                                      download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          # 将输入图像或Numpy数组转换为tensor
                                                                          # 将原始图像的像素值从[0, 255]缩放到 [0, 1]
                                                                          torchvision.transforms.ToTensor(),
                                                                          # 均值-标准差归一化，均值，方差，对于图像的通道，都需要指定一个均值
                                                                          #但这是灰度图，因此只需要一个
                                                                          # x' = x-u/sigma
                                                                          torchvision.transforms.Normalize(
                                                                              (0.1307,), (0.3081,)
                                                                          )
                                                                      ])),
                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),batch_size=batch_size,shuffle=False
)


x,y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
utils.plot_image(x, y, 'image sample')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = relu(h2w3+b3)
        x = self.fc3(x)
        return x

net = Net()
# [w1, b1, w2, b2, w3, b3]
# 设置需要优化的参数，学习率
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []
for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y:[512]
        # [b, feature]
        x = x.view(x.size(0), 28*28)
        # > [b,10]
        out = net(x)
        y_onehot = utils.one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        # 梯度清零，梯度计算基于当前批次数据，确保模型训练过程的正确性和稳定性
        optimizer.zero_grad()
        loss.backward()
        # 更新参数
        # w' = w - lr*grad
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch,batch_idx,loss.item())

utils.plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]


# test
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b, 10]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

# 可视化
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
utils.plot_image(x, pred, 'test')