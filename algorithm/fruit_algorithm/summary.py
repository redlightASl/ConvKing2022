import torch
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 80, backbone = "mobilenetv2").to(device)
    summary(m, input_size=(3, 416, 416))
    
    # mobilenetv2-yolov4 39,062,013

