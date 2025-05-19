#建立数据集的第二步：下载的视频移动至对应分类子文件夹

import os
import shutil
import re


# 设置文件路径
base_path = r"E:/毕业设计/datasets/bili_datasets/raw_data/test/video"
sort_file = r"E:/毕业设计/datasets/bili_datasets/raw_data/sort.txt"

# 创建分类文件夹
categories = [
    "车辆文化", "动物", "二创", "仿妆cos", "搞笑", "鬼畜", "绘画", "科技","科普","美食",
    "明星综合", "人文历史", "三农", "设计·创意", "生活", "时尚潮流", "手办·模玩", "特摄",
    "舞蹈", "音乐", "影视", "游戏", "运动综合", "职业综合", "综合"
]

for category in categories:
    category_path = os.path.join(base_path, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

# 存储移动失败的视频名
failed_videos = []

# 定义一个函数来移除特殊符号和emoji
def clean_title(title):
    # 移除emoji和特殊字符
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()  # 去除多余的空格
    return title

# 处理分类文本文件
with open(sort_file, 'r', encoding='utf-8') as file:
    for line in file:
        if '\t' in line:
            title, category = line.strip().split('\t')
            cleaned_title = clean_title(title)  # 预处理标题
            video_files = os.listdir(base_path)
            matched_file = None
            for video in video_files:
                # 对视频文件名进行预处理并尝试匹配
                cleaned_video = clean_title(video)
                if cleaned_title == cleaned_video[:len(cleaned_title)]:
                    matched_file = video
                    break
            
            if matched_file:
                try:
                    src_path = os.path.join(base_path, matched_file)
                    dest_path = os.path.join(base_path, category, matched_file)
                    shutil.move(src_path, dest_path)
                    print(f"移动成功：{matched_file} -> {category}")
                except Exception as e:
                    print(f"移动失败：{matched_file}，错误：{e}")
                    failed_videos.append(matched_file)
            else:
                print(f"未找到文件：{title}")
                failed_videos.append(title)

# 输出移动失败的视频名
if failed_videos:
    print("\n以下视频移动失败：")
    for video in failed_videos:
        print(video)