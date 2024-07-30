import  torch
from    matplotlib import pyplot as plt

# loss curve 可视化loss下降曲线
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')  #plot(x,y)
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


# visualization of result 可视化识别结果
def plot_image(img, label, name):

    fig = plt.figure() #绘图
    for i in range(6):
        plt.subplot(2, 3, i + 1) # 创建多个子图,2 rows 3 cloums ,i+1 is index
        plt.tight_layout()  #用于自动调整子图参数以提供指定的填充
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none') #draw heatmap
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([]) #不显示横坐标
        plt.yticks([])
    plt.show()

# one_hot encode for label
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth) #返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out