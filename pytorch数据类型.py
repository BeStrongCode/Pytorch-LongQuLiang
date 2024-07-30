import numpy as np
import torch

a = torch.randn(2, 3)
print(a.type())

print(isinstance(a, torch.cuda.FloatTensor))
a = a.cuda()
print(isinstance(a, torch.cuda.FloatTensor))

# dim=0
print(torch.tensor(1.).shape)
# dim=1
print(torch.tensor([1]).shape)

# 将numpy类型转为tensor
a = np.array([2,3.3])
b = torch.from_numpy(a)
print(b)

# 将list转为tensor torch.tensor(list)
print(torch.tensor([2,3.2]))

# 生成0-1范围的数
a = torch.rand(3,3)
print(a)
# 生成与a shape 一样维度的数据
print(torch.rand_like(a))

#生成 N（0.1） 均值为0，方差为1的数据
print(torch.randn(3,3))

# 生成一个shape是【2，3】的数据，初始值都为7
print(torch.full([2,3], 7))

# 生成一个[0,10)的等差数列
print(torch.arange(0,10))

# 等差数列的公差为2
print(torch.arange(0,10,2))

# linspace/logspace
# 等分切割 [0，10] 分为4部分
print(torch.linspace(0,10,steps=4))


# 生成全部是1的矩阵
print(torch.ones(3,3))

# 生成全部是0的矩阵
print(torch.zeros(3,3))

#生成对角线全部是1的矩阵
print(torch.eye(3,4))
print(torch.eye(3)) #3*3 矩阵

# 生成一个从 0到n-1的随机排列
print(torch.randperm(10))

# 根据idx张量中的索引顺序来重新排列原始张量的行
# 当我们执行 a[idx] 时，PyTorch 将按照 idx 中的顺序重新排列 a 的行。具体步骤如下：
# idx 中的第一个值是 1，所以 a[idx][0] 将是 a 的第 1 行。
# idx 中的第二个值是 0，所以 a[idx][1] 将是 a 的第 0 行。
a = torch.rand(2,3)
b = torch.rand(2,2)
idx = torch.randperm(2)
print(a[idx])
print(b[idx])


