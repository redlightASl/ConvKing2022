import torch
from torchsummary import summary
import cv2
import numpy as np
from PIL import Image
import colorsys
from yolo import YOLO
import yolo
from nets.yolo import YoloBody

from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input, resize_image)
from utils.utils_bbox import DecodeBox


anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape = [416, 416]
SAVE_MODEL = 1
anchors_path = 'model_data/yolo_anchors.txt'
classes_path = 'model_data/voc_classes.txt'
model_path = 'model_data/yolov4_mobilenet_v2_voc.pth'


class_names, num_classes  = get_classes(classes_path)
anchors, num_anchors = get_anchors(anchors_path)
bbox_util = DecodeBox(anchors, num_classes, (input_shape[0], input_shape[1]), anchors_mask)


hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))


if __name__ == "__main__":
    # net = YoloBody(anchors_mask, num_classes, backbone = "mobilenetv2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net.load_state_dict(torch.load(model_path, map_location=device))

    # net = net.eval()

    net = YoloBody(anchors_mask, 80, backbone = "mobilenetv2").to(device)
    summary(net, input_size=(3, 416, 416))

    """
    保存模型
    """
    if(SAVE_MODEL):
        torch.save(net.state_dict(), "./my_model.pt")  # 保存模型参数
        input_names = ['inputs']  # 输入的名字
        output_names = ['boxes', 'confs']  # 输出的名字
        # once_batch_size多少决定使用onnx模型时一次处理多少图片
        once_batch_size = 1
        # 输入图片的通道,高,宽
        channel = 3
        height = 416
        width = 416
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


