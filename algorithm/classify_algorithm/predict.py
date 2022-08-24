import time
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from nets.classify_net import MobileNetV2
from torch.nn import functional as F

classes = [
    'apple', 'banana', 'grape', 'kiwi', 'mango', 'orange', 'pear', 'pitaya'
]

transform = transforms.Compose([
    transforms.Resize((96,96)),  #调整图片大小
    transforms.ToTensor()  #转换成Tensor，将图片取值范围转换成0-1之间，将channel置前
])

def to_numpy(tensor):
    """tensor将转换成numpy模型"""
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
    model = MobileNetV2(in_dim=3, num_classes=8)
    mode = "video"
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = transform(image)
                r_image = r_image.unsqueeze(0)
                weight = torch.load('mobilenet_model.pt')
                model.load_state_dict(weight)
                model.eval()
                outputs = model(r_image)
                print(outputs)
                data_softmax = F.softmax(outputs, dim=1).squeeze(dim=0).detach().numpy()
                index = data_softmax.argmax()
                print(classes[index])
    elif mode == "video":
        model = MobileNetV2(in_dim=3, num_classes=8)
        weight = torch.load('mobilenet_model.pt')
        model.load_state_dict(weight)
        model.eval()

        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("Video Read Error!")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            r_image = Image.fromarray(np.uint8(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            r_image = transform(r_image)
            r_image = r_image.unsqueeze(0)

            outputs = model(r_image)

            # print(outputs)
            data_softmax = F.softmax(outputs, dim=1).squeeze(dim=0).detach().numpy()
            index = data_softmax.argmax()

            # print(classes[index])
            frame = cv2.putText(frame, classes[index], (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            
            cv2.imshow("video",frame)

            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
    else:
        raise AssertionError("Please specify the correct mode: 'predict'or 'video'!")
