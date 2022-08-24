import torch
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
from nets.classify_net import MobileNetV2

classes = [
    'apple', 'banana', 'grape', 'kiwi', 'mango', 'orange', 'pear', 'pitaya'
]

transform = transforms.Compose([
    transforms.Resize((28,28)),  #调整图片大小
    transforms.ToTensor()  #转换成Tensor，将图片取值范围转换成0-1之间，将channel置前
])

model = MobileNetV2(in_dim=3, num_classes=8)

image = Image.open("./data/ConvKing/kiwi/kiwi_10.png")

r_image = transform(image)
r_image = r_image.unsqueeze(0)
weight = torch.load('mobilenet_model.pt')
model.load_state_dict(weight)
model.eval()
outputs = model(r_image)
# print(outputs)
data_softmax = F.softmax(outputs, dim=1).squeeze(dim=0).detach().numpy()
index = data_softmax.argmax()
print(classes[index])
