import cv2
import numpy as np

file_name = './img.jpg' # 要输入的图片
coe = "./coe_out/image.coe"  # 要输出的coe文件

file = open(coe, "w")
file.write("memory_initialization_radix = 16;\n") # 数据以hex格式存储
file.write("memory_initialization_vector =\n")

img = cv2.imread(file_name)
# cv2.imshow("img", img)
# cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGR565)
img = np.array(img, dtype=np.uint16)

print(img.shape)

height = img.shape[0]
width = img.shape[1]

for i in range(height):
    for j in range(width):
        # 将图片转换为coe格式
        hex_a = img[i][j][0] | (img[i][j][1]<<8)
        file.write(str(hex(hex_a)).lstrip('0x').rjust(4, '0') + "\n")

