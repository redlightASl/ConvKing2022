import glob
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

TRAIN_DATASET_RATE = 0.8 # 80% as train
classes = ['apple','banana','grape','kiwi','mango','orange','pear','pitaya']

class ConvKing_dataset(torch.utils.data.Dataset): #创建data.Dataset子类Mydataset来创建输入
    def __init__(self, imgs_path): # 传入数据目录路径
        self._imgs_path = imgs_path

    def __getitem__(self, index): # 根据索引下标获得相应的图片
        _img = self._imgs_path[index]
        return _img

    def __len__(self): # 返回整个数据目录下所有文件的个数
        return len(self._imgs_path)

#使用glob获取数据图片的所有路径
all_imgs_path = glob.glob(r'.\data\ConvKing\*\*.png') #数据目录路径
print("========all raw images:========")
for var in all_imgs_path:
    print(var)
print("========all raw images show done!========")

#创建数据集对象
fruit_dataset = ConvKing_dataset(all_imgs_path)
print("total image number:", len(fruit_dataset)) #返回文件夹中图片总个数
# print(fruit_dataset[12:15]) #切片显示第12至15张图片的路径
fruit_datalodaer = torch.utils.data.DataLoader(fruit_dataset, batch_size=5) #每次迭代时返回五个数据
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
print(all_labels) #得到所有标签

# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Resize((96,96)), #调整图片大小
    transforms.ToTensor() #转换成Tensor，将图片取值范围转换成0-1之间，将channel置前
])

print("========Test Done!========\n")

class ConvKingData(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self._imgs = img_paths
        self._labels = labels
        self._transforms = transform

    def __getitem__(self, index): #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        _img = self._imgs[index]
        label = self._labels[index]
        pil_img = Image.open(_img)
        data = self._transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self._imgs)

BATCH_SIZE = 10
fruit_dataset = ConvKingData(all_imgs_path, all_labels, transform)
fruit_datalodaer = torch.utils.data.DataLoader(
                            fruit_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True
)
imgs_batch, labels_batch = next(iter(fruit_datalodaer))
print(imgs_batch.shape)

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i+1)
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

train_ds = ConvKingData(train_imgs, train_labels, transform) #TrainSet TensorData
test_ds = ConvKingData(test_imgs, test_labels, transform) #TestSet TensorData
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) #TrainSet Labels
test_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) #TestSet Labels

print("========Package Dataset Done!========")

