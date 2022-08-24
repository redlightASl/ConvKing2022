import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.quantization

SAVE_MODEL=1
SAVE_QMODEL=1

class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=[3,3], stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=200),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=10),
            nn.ReLU()
        )

    def forward(self, x):
        # x = self.conv1(x)
        x = x.view(-1, 28*28)
        # x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

"""
预览数据集
"""
batch_size = 64
# 装载训练集
train_dataset=datasets.MNIST(root='./num/',train=True,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)
# 装载测试集
test_dataset=datasets.MNIST(root='./num/',train=False,transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

"""
训练网络
"""
LR = 0.001
device = torch.device('cuda')
net = NewNet().to(device)

criterion = nn.CrossEntropyLoss()  # 损失函数使用交叉熵
optimizer = optim.Adam(net.parameters(), lr=LR)  # 优化函数使用 Adam 自适应优化算法

epoch = 1

if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

    """
    测试网络
    """
    net.eval()  # 将模型变换为测试模式
    correct = 0
    total = 0

    for data_test in test_loader:
        images, labels = data_test
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        output_test = net(images)
        # print(output_test)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct: ", correct)
    print("Test acc: {0}".format(correct.item() / len(test_dataset)))

    """
    保存模型
    """
    # q_backend = "fbgemm"  # qnnpack  or fbgemm
    # torch.backends.quantized.engine = q_backend
    # qconfig = torch.quantization.get_default_qconfig(q_backend)
    quantized_model = torch.quantization.quantize_dynamic(
        net, {nn.fc1, nn.fc2}, dtype=torch.qint8
    ) # 量化
    print(quantized_model)

    if(SAVE_MODEL):
        torch.save(net.state_dict(), "./my_model.pt")  # 保存模型参数
        input_names = ['inputs']  # 输入的名字
        output_names = ['outputs']  # 输出的名字
        # once_batch_size多少决定使用onnx模型时一次处理多少图片
        once_batch_size = 1
        # 输入图片的通道,高,宽
        channel = 1
        height = 28
        width = 28
        dummy_input = torch.randn(once_batch_size,
                                channel,
                                height,
                                width,
                                requires_grad=True)

        net.cpu()  # 保存为onnx之前，先将model转为CPU模式
        torch.onnx.export(net, (dummy_input),
                        "./my_model.onnx",
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names)
    
    if(SAVE_QMODEL):
        torch.save(quantized_model.state_dict(), "./my_qmodel.pt")  # 保存模型参数
        input_names = ['inputs']  # 输入的名字
        output_names = ['outputs']  # 输出的名字
        # once_batch_size多少决定使用onnx模型时一次处理多少图片
        once_batch_size = 1
        # 输入图片的通道,高,宽
        channel = 1
        height = 28
        width = 28
        dummy_input = torch.randn(once_batch_size,
                                channel,
                                height,
                                width,
                                requires_grad=True)

        quantized_model.cpu()  # 保存为onnx之前，先将model转为CPU模式
        torch.onnx.export(quantized_model, (dummy_input),
                        "./my_model.onnx",
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names)

