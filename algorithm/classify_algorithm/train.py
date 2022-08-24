import glob
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from nets.classify_net import MobileNetV2

TRAIN_DATASET_RATE = 0.8  # 80% as train
BATCH_SIZE = 10
IMG_SIZE = (96, 96)
classes = [
    'apple', 'banana', 'grape', 'kiwi', 'mango', 'orange', 'pear', 'pitaya'
]
learning_rate = 1e-3
num_epochs = 20


class ConvKing_dataset(torch.utils.data.Dataset
                       ):  #创建data.Dataset子类Mydataset来创建输入
    def __init__(self, imgs_path):  # 传入数据目录路径
        self._imgs_path = imgs_path

    def __getitem__(self, index):  # 根据索引下标获得相应的图片
        _img = self._imgs_path[index]
        return _img

    def __len__(self):  # 返回整个数据目录下所有文件的个数
        return len(self._imgs_path)


#使用glob获取数据图片的所有路径
all_imgs_path = glob.glob(r'.\data\ConvKing\*\*.png')  #数据目录路径
print("========all raw images:========")
for var in all_imgs_path:
    print(var)
print("========all raw images show done!========")

#创建数据集对象
fruit_dataset = ConvKing_dataset(all_imgs_path)
print("total image number:", len(fruit_dataset))  #返回文件夹中图片总个数
# print(fruit_dataset[12:15]) #切片显示第12至15张图片的路径
fruit_datalodaer = torch.utils.data.DataLoader(fruit_dataset,
                                               batch_size=5)  #每次迭代时返回五个数据
print(next(iter(fruit_datalodaer)))

#为图片设置类别
classes_to_id = dict((c, i) for i, c in enumerate(classes))
print(classes_to_id)
id_to_classes = dict((v, k) for k, v in classes_to_id.items())
print(id_to_classes)

all_labels = []
for img in all_imgs_path:
    # 设置每个img的类别
    for i, c in enumerate(classes):
        if c in img:
            all_labels.append(i)
print(all_labels)  #得到所有标签

# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  #调整图片大小
    transforms.ToTensor()  #转换成Tensor，将图片取值范围转换成0-1之间，将channel置前
])

print("========Test Done!========\n")


class ConvKingData(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self._imgs = img_paths
        self._labels = labels
        self._transforms = transform

    def __getitem__(self, index):  #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        _img = self._imgs[index]
        label = self._labels[index]
        pil_img = Image.open(_img)
        data = self._transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self._imgs)

fruit_dataset = ConvKingData(all_imgs_path, all_labels, transform)
fruit_datalodaer = torch.utils.data.DataLoader(fruit_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
imgs_batch, labels_batch = next(iter(fruit_datalodaer))
print(imgs_batch.shape)

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i + 1)
    plt.title(id_to_classes.get(label.item()))
    plt.imshow(img)
plt.show()

#划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))
all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

s = int(len(all_imgs_path) * TRAIN_DATASET_RATE)
print(s)

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_imgs_path[s:]

train_ds = ConvKingData(train_imgs, train_labels, transform)
test_ds = ConvKingData(test_imgs, test_labels, transform)
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=True)  #训练集
test_loader = torch.utils.data.DataLoader(train_ds,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)  #测试集

print("========Package Dataset Done!========")
print("========Training Start!========")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = MobileNetV2(in_dim=3, num_classes=8).to(device)

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
            labels = torch.tensor(labels,dtype=torch.long)
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
            .format(epoch, train_loss_sum / BATCH_SIZE,
                    train_correct / train_accuracy_total, test_acc))
    print('========Training Finish!========')
    return model


def save_model(net):
    net.cpu()  # 保存为onnx之前，先将model转为CPU模式
    torch.save(net.state_dict(), "./mobilenet_model.pt")  # 保存模型参数
    input_names = ['inputs']  # 输入的名字
    output_names = ['outputs']  # 输出的名字
    # once_batch_size多少决定使用onnx模型时一次处理多少图片
    once_batch_size = 1
    # 输入图片的通道,高,宽
    channel = 3
    height = IMG_SIZE[0]
    width = IMG_SIZE[1]
    dummy_input = torch.randn(once_batch_size,
                              channel,
                              height,
                              width,
                              requires_grad=True)
    torch.onnx.export(net, (dummy_input),
                      "./mobilenet_model.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = train()
    save_model(model)