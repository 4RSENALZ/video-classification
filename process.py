# 数据预处理
# 视频帧图像提取（10帧提取一次）
# 视频音频提取

import os
import cv2
import subprocess
import re
from tqdm import tqdm

# 设置路径
video_source_path = r"E:/毕业设计/datasets/bili_datasets/raw_data/test/video"
frames_output_base = r"E:/毕业设计/datasets/bili_datasets/processed/frames/test"
audio_output_base = r"E:/毕业设计/datasets/bili_datasets/raw_data/test/audio"

# 分类列表
categories = [
    "车辆文化", "动物", "二创", "仿妆cos", "搞笑", "鬼畜", "绘画", "科技", "科普", "美食",
    "明星综合", "人文历史", "三农", "设计·创意", "生活", "时尚潮流", "手办·模玩", "特摄",
    "舞蹈", "音乐", "影视", "游戏", "运动综合", "职业综合", "综合"
]

# 创建输出目录
os.makedirs(frames_output_base, exist_ok=True)
os.makedirs(audio_output_base, exist_ok=True)

# 存储失败的视频名
failed_videos = []

def clean_title(title):
    # 移除 BV 号（假设 BV 号格式为 [BV...]）
    title = re.sub(r'\[BV[^\]]*\]', '', title)
    # 移除特殊符号，保留中文、字母、数字和空格
    title = re.sub(r'[^\w\u4e00-\u9fa5\s]', '', title)
    # 将空格和多个下划线替换为单个下划线
    title = re.sub(r'[\s_]+', '_', title)
    # 去除标题末尾的下划线
    title = title.rstrip('_')
    # 截断过长标题（保留前50字符）
    return title[:50]

def extract_frames(video_path, output_dir, video_name):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件：{video_path}")
            return -1

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
        cleaned_video_name = clean_title(video_name)  # 清理标题

        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        print(f"保存路径: {output_dir}")  # 调试信息

        with tqdm(total=frame_count, desc=f"提取帧：{cleaned_video_name}", unit="帧") as pbar:
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                #每10帧提取一次图像
                if count % 10==0:
                    # 完整路径
                    frame_file = os.path.join(output_dir, f"{cleaned_video_name}{count:04d}.jpg")

                    try:
                        _, encoded_image = cv2.imencode('.jpg', frame)
                        encoded_image.tofile(frame_file)
                    except Exception as e:
                        print(f"保存帧失败：{frame_file}，错误：{e}")

                count += 1
                pbar.update(1)  # 更新进度条

        cap.release()
        return count  # 返回提取的总帧数

    except Exception as e:
        print(f"提取帧失败：{video_name}，错误信息：{e}")
        return -1

# 音频提取函数
def extract_audio(video_path, output_dir, video_name):
    try:
        cleaned_video_name = clean_title(video_name)  # 清理标题
        audio_file = os.path.join(output_dir, f"{cleaned_video_name}.wav")
        # 使用 ffmpeg 提取音频
        command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_file}\" -y"
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_file

    except Exception as e:
        print(f"提取音频失败：{video_name}，错误信息：{e}")
        return None  # 返回 None 表示失败
    
if __name__ == "__main__":
    for category in categories:
        category_path = os.path.join(video_source_path, category)
        if not os.path.exists(category_path):
            print(f"分类目录不存在，跳过：{category}")
            continue

        videos = [v for v in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, v))]

        for video in videos:
            video_path = os.path.join(category_path, video)
            video_name, _ = os.path.splitext(video)
            cleaned_video_name = clean_title(video_name)

            frames_output_dir = os.path.join(frames_output_base, category, cleaned_video_name)
            audio_output_category_dir = os.path.join(audio_output_base, category)
            audio_file_path = os.path.join(audio_output_category_dir, f"{cleaned_video_name}.wav")

            # 检查帧图像是否已存在（存在至少一帧图像文件）
            frame_exists = os.path.exists(frames_output_dir) and len(os.listdir(frames_output_dir)) > 0

            # 检查音频是否已存在
            audio_exists = os.path.exists(audio_file_path)

            if frame_exists and audio_exists:
                print(f"跳过已处理视频：{video_name}")
                continue

            print(f"处理视频：{video_path}")

            if not frame_exists:
                frame_count = extract_frames(video_path, frames_output_dir, cleaned_video_name)
                if frame_count == -1:
                    failed_videos.append(f"{category}/{video_name}")
                    continue
                print(f"已提取 {frame_count} 帧图像，存放路径：{frames_output_dir}")
            else:
                print(f"已存在帧图像，跳过提取：{cleaned_video_name}")

            if not audio_exists:
                os.makedirs(audio_output_category_dir, exist_ok=True)
                audio_file = extract_audio(video_path, audio_output_category_dir, cleaned_video_name)
                if audio_file is None:
                    failed_videos.append(f"{category}/{video_name}")
                    continue
                print(f"音频文件保存为：{audio_file}")
            else:
                print(f"已存在音频文件，跳过提取：{cleaned_video_name}")

def process_single_video(video_path, output_base_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cleaned_video_name = clean_title(video_name)

    # 创建图像和音频输出路径
    frame_output_dir = os.path.join(output_base_dir, "frames", cleaned_video_name)
    audio_output_dir = os.path.join(output_base_dir, "audio")
    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    # -------- 提取帧 --------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return None, None, cleaned_video_name

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=frame_count, desc=f"提取帧: {cleaned_video_name}", unit="帧") as pbar:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % 10 == 0:
                frame_file = os.path.join(frame_output_dir, f"{cleaned_video_name}{count:04d}.jpg")
                _, encoded_image = cv2.imencode('.jpg', frame)
                encoded_image.tofile(frame_file)
            count += 1
            pbar.update(1)
    cap.release()

    # -------- 提取音频 --------
    audio_file = os.path.join(audio_output_dir, f"{cleaned_video_name}.wav")
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_file}" -y'
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return frame_output_dir, audio_file, cleaned_video_name
