#划分数据集（训练集70、测试集15、验证集15）

import os
import random
import shutil


# 设置源目录和验证集、测试集的目标目录
source_base = r"E:/毕业设计/datasets/bili_datasets/raw_data/video"
val_base = r"E:/毕业设计/datasets/bili_datasets/raw_data/val/video"
test_base = r"E:/毕业设计/datasets/bili_datasets/raw_data/test/video"

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 随机种子（用于结果可复现）
random.seed(42)

# 创建验证集和测试集的目标目录
for base in [val_base, test_base]:
    if not os.path.exists(base):
        os.makedirs(base)

# 遍历每个分类文件夹
categories = [d for d in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, d))]

for category in categories:
    category_path = os.path.join(source_base, category)
    videos = [v for v in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, v))]
    
    # 打乱视频顺序
    random.shuffle(videos)
    
    # 计算划分数量
    total_videos = len(videos)
    train_count = int(total_videos * train_ratio)
    val_count = int(total_videos * val_ratio)
    
    # 划分数据集
    train_videos = videos[:train_count]
    val_videos = videos[train_count:train_count + val_count]
    test_videos = videos[train_count + val_count:]
    
    # 将验证集视频文件复制到目标目录
    for video in val_videos:
        src_path = os.path.join(category_path, video)
        dest_path = os.path.join(val_base, video)
        shutil.move(src_path, dest_path)  # 使用复制
        print(f"已将 {video} 移动到验证集 (val)/{category}/")

    # 将测试集视频文件复制到目标目录
    for video in test_videos:
        src_path = os.path.join(category_path, video)
        dest_path = os.path.join(test_base, video)
        shutil.move(src_path, dest_path)  # 使用复制
        print(f"已将 {video} 移动到测试集 (test)/{category}/")

print("验证集和测试集划分完成！")
