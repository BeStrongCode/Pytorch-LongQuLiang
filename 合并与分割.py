import torch

# 非cat维度必须保持一致
a = torch.rand(4,32,8)
b = torch.rand(5,32,8)

print(torch.cat([a,b],dim=0).shape)

a1 = torch.rand(4,3,32,32)
a2 = torch.rand(5,3,32,32)
print(torch.cat([a1,a2],dim=0).shape)
a2 = torch.rand(4,1,32,32)
print(torch.cat([a1,a2],dim=1).shape)

a1 = torch.rand(4,3,16,32)
a2 = torch.rand(4,3,16,32)
print(torch.cat([a1,a2],dim=2).shape)

# stack 会插入一个新的维度
# 以下例子适合用stack而不适合cat
# 一班 32个人 8门课
# 二班 32个人 8门课
# 将两个班级合成一张表，此时适合用stack增添一个新的维度 如果用cat 产生 [64,8] 不符合实际需求
a = torch.rand(32,8)
b = torch.rand(32,8)
print(torch.stack([a,b],dim=0).shape)


# split 根据长度拆分
c = torch.stack([a,b],dim=0)
aa, bb = c.split(1,dim=0) # 拆分成每个单元长度都为1
print(aa.shape,bb.shape)
c = torch.rand(3,32,8)
aa, bb = c.split([2,1],dim=0) # 第一个单元长度为2，第二个单元长度为1
print(aa.shape,bb.shape)

# chunk 按数量拆分
b = torch.rand(2,32,8)
aa, bb = b.chunk(2,dim=0) # 在第0维上拆分成两块