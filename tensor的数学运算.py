import torch

a = torch.rand(3,4)
b = torch.rand(4)
# add
print(a+b)
print(torch.add(a,b))

# 来查看一下 add 操作和 + 是否一致
print(torch.all(torch.eq(a+b,torch.add(a,b))))

# 矩阵相乘 matmul
# 有三种形式:
# torch.mm()
# torch.matmul()
# @
b = torch.ones(3,3)
a = torch.ones(3,3)
print(torch.matmul(a,b))

# 对于二维以上的矩阵 torch.mm无法运算，会报错

a = torch.rand(4,3,28,64)
b = torch.rand(4,3,64,32)
print(torch.matmul(a,b).shape)  # [4,3,28,32]  取后面两维进行运算，前面保持不变

b = torch.rand(4,1,64,32)
print(torch.matmul(a,b).shape) # [4,3,28,32] 使用了broadcasting

b = torch.rand(4,64,32) #这个向量无法broadcasting，[4,64,32] => [1,4,64,32] ? [4,3,28,64] 4!=3

# power 次方运算
a = torch.full([2,2],3)
print(a.pow(2))
print(a**2)

# Exp log
a = torch.exp(torch.ones(2,2))
print(a)
print(torch.log(a)) # 默认以e为底

# Approximation 近似值
a = torch.tensor(3.14)
print(a.floor()) # 向下取整
print(a.ceil()) # 向上取整
print(a.trunc()) # 裁剪出整数部分
print(a.frac()) # 裁剪出小数部分

a = torch.tensor(3.499)
print(a.round()) # 四舍五入
a = torch.tensor(3.5)
print(a.round())