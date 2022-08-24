import torch
from torchvision import datasets, transforms
# from nets.mobilenet_v2 import MobileNetV2
from nets.classify_net import MobileNetV2


# 训练集和测试集的数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])
test_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])

# 下载训练集
train_dataset = datasets.MNIST(
    root='./data/',  # 用于指定数据集在下载之后的存放路径
    train=True,  # 指定在数据集下载完成后需要载入的那部分数据:True 载入训练集；False 载入测试集
    transform=train_transform,  # 用于指定导入数据集需要对数据进行哪种变化操作
    download=True)  # 需要程序自动下载
# 下载测试集
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=test_transform,
                              download=True)

batch_size = 64  # batch大小

# 装载训练集
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,  # 用于指定我们载入的数据集名称
    batch_size=batch_size,  # 设置每个包中的图片数据个数
    shuffle=True,
    drop_last=True)  # 在装载的过程会将数据随机打乱顺序并进打包
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

learning_rate = 1e-3
num_epochs = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = MobileNetV2(in_dim=1, num_classes=10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


def evaluate_accuracy(data_iter, model):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for images, labels in data_iter:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).cpu().sum().numpy()
    return correct / total


def train(data_loader=train_loader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          epochs=num_epochs,
          device=device):
    for epoch in range(epochs):
        print('current epoch = {}'.format(epoch))
        for i, (images, labels) in enumerate(data_loader):
            train_accuracy_total = 0
            train_correct = 0
            train_loss_sum = 0
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)  # 计算模型的损失
            optimizer.zero_grad()  # 在做反向传播前先清除网络状态
            loss.backward()  # 损失值进行反向传播
            optimizer.step()  # 参数迭代更新

            train_loss_sum += loss.item(
            )  # item()返回的是tensor中的值，且只能返回单个值（标量），不能返回向量，使用返回loss等
            _, predicts = torch.max(outputs.data, dim=1)  # 输出10类中最大的那个值
            train_accuracy_total += labels.size(0)
            train_correct += (predicts == labels).cpu().sum().item()
        test_acc = evaluate_accuracy(test_loader, model)
        print(
            'epoch:{0},   loss:{1:.4f},   train accuracy:{2:.3f},  test accuracy:{3:.3f}'
            .format(epoch, train_loss_sum / batch_size,
                    train_correct / train_accuracy_total, test_acc))
    print('------------finish training-------------')
    return model

def save_model(net):
    torch.save(net.state_dict(), "./mobilenet_mnist_model.pt")  # 保存模型参数
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
                      "./mobilenet_mnist.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = train()
    save_model(model)