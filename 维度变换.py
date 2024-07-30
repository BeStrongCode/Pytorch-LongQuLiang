import torch

# 常用api
# view/reshape
# squeeze/unsqueeze
# transpose/t/permute 转置
# expand/repeat 维度扩展

# 数据的存储顺序非常重要，还原的时候不能打乱
a = torch.rand(4,1,28,28) # B,C,H,W
print(a.shape)
a = a.view(4,28*28)
print(a.shape)
b = a.view(4,28,28,1) # B H W C 这会造成数据污染

# squeeze and unsqueeze
# 只改变数据的理解方式，不改变数据原本的位置
c = a.unsqueeze(0) # 在o维度前面插入一个维度 4,1,28,28 => 1,4,1,28,28
print(a.unsqueeze(-1).shape) #在最后一个维度插入一个维度 4,1,28,28 => 4,1,28,28,1

# for example
bias = torch.rand(32) #给每个channel加一个偏置 如何f+b, 改变b的维度
f = torch.rand(4,32,14,14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0) # 1 32 1 1

print(b.squeeze().shape) # 1 32 1 1 => 32
# 将第0维度挤压
print(b.squeeze(0).shape) # 1 32 1 1 => 32 1 1
# 将第-1维挤压
print(b.squeeze(-1).shape) # 1 32 1 1 => 1 32 1
# 将第1维挤压，但是第1维不为1，所以无法挤压 结果不变
print(b.squeeze(1).shape) # 1 32 1 1 => 1 32 1 1

# exapnd/ repeat 维度扩展
# expand: broadcasting  只改变理解方式并没有增加数据
# repeat: memory copied 实实在在的增加了数据
b = torch.rand(1, 32, 1, 1)
# 仅限于将维度从1 - N 如果本身维度不为1会报错，无法复制
print(b.expand(4,32,14,14).shape) # 1 32 1 1 => 4 32 14 14

# repeat 每一个维度要重复的次数
print(b.repeat(4,32,1,1).shape) # 1 32 1 1 => 4 32*32 1 1

# transpose
a = torch.rand(4,3,32,32)
a1 = a.transpose(1,3).view(4,3*32*32).view(4,3,32,32) # bchw => bwhc => bcwh 维度顺序被打乱了 报错

# 在 PyTorch 中，transpose 操作会改变张量的维度顺序，但不会改变张量在内存中的存储方式。这样，虽然张量看起来像是已经转置了，但实际上其在内存中的布局没有改变。这可能会导致后续的某些操作（如 view）无法正确执行，因为这些操作要求张量在内存中是连续存储的。
# 使用 contiguous() 可以重新排列张量的数据，使其在内存中变为连续存储。这对于确保后续的 view 操作能够正确地进行是必要的。
a1 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32) # bchw => bwhc => bcwh 维度顺序打乱，
a1 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3) #bchw => bwhc => bchw 正确

# permute
# transpose只能交换两个，permute可以交换多个
a1 = a.permute(0,2,3,1)