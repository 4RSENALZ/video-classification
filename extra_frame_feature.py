# 提取视频帧图像的特征向量

import os
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd

# 文件路径
frames_base_path = r"E:/毕业设计/datasets/bili_datasets/processed/frames/test"
output_base_path = r"E:/毕业设计/datasets/bili_datasets/features/frame_resnet/test"

# 创建特征输出目录
os.makedirs(output_base_path, exist_ok=True)

# 预训练 ResNet
model = models.resnet34(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉fc层，模型的输出是最后一层卷积特征图的全局平均池化结果，表示输入图像的特征向量。
model.eval()

# 定义预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 解决 OpenCV 不能读取中文路径的问题
def read_image(frame_path):
    return cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 处理全白/全黑图像
def is_blank_image(img, threshold=10):
    #判断图像是否是全白或全黑
    #`threshold` 设定像素方差的最低值（越小越严格）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray) < threshold  # 方差过低表示几乎无变化

# 图像预处理函数
def preprocess_image(img):
    #对图像进行预处理：
    #1. 移除黑边
    #2. 调整长宽比
    #3. 标准化到224*224的比例（用了黑色填充，也可以试试镜像填充之类的）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    
    # 解决 ZeroDivisionError
    if w == 0 or h == 0:
        return None  # 直接丢弃该帧

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

    return transform(img_padded)

if __name__ == "__main__":
    # 处理视频帧
    categories = os.listdir(frames_base_path)
    for category in categories:
        category_path = os.path.join(frames_base_path, category)
        if not os.path.isdir(category_path):
            continue

        category_output_path = os.path.join(output_base_path, category)
        os.makedirs(category_output_path, exist_ok=True)

        videos = os.listdir(category_path)
        for video in videos:
            video_path = os.path.join(category_path, video)
            if not os.path.isdir(video_path):
                continue

            output_csv_path = os.path.join(category_output_path, f"{video}.csv")

            # 跳过已处理的视频
            if os.path.exists(output_csv_path):
                print(f"跳过已处理视频：{video}")
                continue

            print(f"正在处理视频：{video}")
            features = []

            frames = sorted(os.listdir(video_path))
            for frame in frames:
                frame_path = os.path.join(video_path, frame)

                if not os.path.exists(frame_path):
                    print(f"文件不存在：{frame_path}")
                    continue

                img = read_image(frame_path)
                if img is None:
                    print(f"无法读取帧图像：{frame_path}")
                    continue

                # 跳过全白/全黑帧
                if is_blank_image(img):
                    print(f"跳过全白/全黑帧：{frame_path}")
                    continue

                img_tensor = preprocess_image(img)
                if img_tensor is None:
                    print(f"跳过无效帧（黑边去除后尺寸为0）：{frame_path}")
                    continue

                img_tensor = img_tensor.unsqueeze(0)

                with torch.no_grad():
                    feature_vector = model(img_tensor)
                    feature_vector = feature_vector.squeeze().numpy()

                frame_name = f"{video}_{frame}"
                frame_name = frame_name.encode("utf-8").decode("utf-8")  # 解决乱码

                features.append([frame_name] + feature_vector.tolist())

            if features:
                columns = ["frame_name"] + [f"feature_{i}" for i in range(feature_vector.shape[0])]
                df = pd.DataFrame(features, columns=columns)

                df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

                print(f"特征保存至：{output_csv_path}")
            else:
                print(f"{video} 没有有效帧，跳过保存。")
                
#写了一个函数来提取单个视频的特征向量，有点冗余了，主要是用来在demo里调用
#如果要批量处理，直接python extra_frame_feature.py就行了
def extract_single_video_feature(frame_dir, output_dir, video_name):
    """
    给定帧图像文件夹路径，提取所有帧图像的特征向量，保存为CSV，并返回特征向量。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"{video_name}.csv")

    # 如果已存在 CSV 文件，则读取并返回特征向量
    if os.path.exists(output_csv_path):
        print(f"跳过已处理视频：{video_name}（已存在特征文件）")
        df = pd.read_csv(output_csv_path)
        if "frame_name" in df.columns:
            df = df.drop(columns=["frame_name"])
        return df.values

    print(f"正在提取帧图像特征：{video_name}")
    features = []

    frame_list = sorted(os.listdir(frame_dir))
    for frame_file in frame_list:
        frame_path = os.path.join(frame_dir, frame_file)

        if not os.path.exists(frame_path):
            print(f"找不到帧图像：{frame_path}")
            continue

        img = read_image(frame_path)
        if img is None:
            print(f"无法读取帧图像：{frame_path}")
            continue

        if is_blank_image(img):
            print(f"跳过全白/全黑帧：{frame_path}")
            continue

        img_tensor = preprocess_image(img)
        if img_tensor is None:
            print(f"无法预处理帧图像：{frame_path}")
            continue

        img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度
        with torch.no_grad():
            feature_vector = model(img_tensor).squeeze().numpy()

        features.append(feature_vector)

    # 保存并返回
    if features:
        features = np.array(features)
        columns = [f"feature_{i}" for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=columns)
        df.insert(0, "frame_name", frame_list[:len(df)])  # 插入帧文件名
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"图像特征保存至：{output_csv_path}")
        return features
    else:
        print(f"没有可用帧图像特征：{frame_dir}")
        return None