import torch
import onnxruntime as ort
import numpy as np
import cv2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


print("------预处理------")
img = cv2.imread("testbench/8_2214_23215.jpg")
img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, dsize=(28, 28))
cv2.imshow("gray", img)
# cv2.waitKey(0)

img = torch.from_numpy(img)  #uint8
img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
img = img.float().div(255)  #float32
# print(img)

print("------加载模型------")
ort_session = ort.InferenceSession("my_model.onnx",
                                   providers=['CPUExecutionProvider'])

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}

print("------运行------")
ort_outs = ort_session.run(None, ort_inputs)  #推理
# _,num = torch.max(torch.tensor(ort_outs),1,1)
# print(num)
ort_outs = np.array(ort_outs)
print(ort_outs[0][0])  #输出结果

# find_number=max(ort_outs[0][0])
# print(find_number)

find_number=np.where(ort_outs[0][0]==np.max(ort_outs[0][0]))
print(find_number)