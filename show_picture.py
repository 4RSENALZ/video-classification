# 测试图像预处理效果

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 解决 OpenCV 不能读取中文路径的问题
def read_image(image_path):
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    # 处理无效帧
    if w == 0 or h == 0:
        return None, None  

    img_cropped = img[y:y+h, x:x+w]

    # 调整尺寸
    scale = 224 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_cropped, (new_w, new_h))

    # 填充到 224x224
    top = (224 - new_h) // 2
    bottom = 224 - new_h - top
    left = (224 - new_w) // 2
    right = 224 - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img_cropped, img_padded

# 随便选了一张图
test_image_path = r"E:/毕业设计/datasets/bili_datasets/processed/frames/train/仿妆cos/粉丝想看我爸cos森鸥外_我给他找了个新女儿/粉丝想看我爸cos森鸥外_我给他找了个新女儿0790.jpg"

img = read_image(test_image_path)

if img is None:
    print(f"无法读取图像：{test_image_path}")
else:
    img_cropped, img_padded = preprocess_image(img)

    if img_cropped is None or img_padded is None:
        print("图像处理后无效，可能是全黑/全白帧")
    else:
        # 显示原图、去黑边后的图像和最终填充的图像
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
        plt.title("cropped")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB))
        plt.title("padding 224x224")
        plt.axis("off")

        plt.show()
