# 把音频转换为频谱图（完全没用到，extra_audio_feature里直接一个函数解决）

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import numpy as np

# 设置支持中文的字体
def set_chinese_font():
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        font = FontProperties(fname=font_path)
        rcParams['font.family'] = font.get_name()
    except Exception as e:
        print(f"未找到中文字体，请检查字体路径：{e}")

# 调用中文字体设置
set_chinese_font()

# 文件路径
audio_base_path = r"E:/毕业设计/datasets/bili_datasets/raw_data/train/audio"
output_base_path = r"E:/毕业设计/datasets/bili_datasets/processed/spectrograms/train"

# 创建输出目录
os.makedirs(output_base_path, exist_ok=True)

# 分类列表
categories = os.listdir(audio_base_path)

# 记录失败音频的数量
failed_audio_count = 0
failed_audio_list = []

for category in categories:
    category_path = os.path.join(audio_base_path, category)
    if not os.path.isdir(category_path):
        continue

    # 创建分类的频谱图输出目录
    category_output_path = os.path.join(output_base_path, category)
    os.makedirs(category_output_path, exist_ok=True)

    # 遍历音频文件
    audio_files = os.listdir(category_path)
    for audio_file in audio_files:
        audio_path = os.path.join(category_path, audio_file)

        # 检查是否是 WAV 文件
        if not audio_file.endswith(".wav"):
            print(f"跳过非 WAV 文件：{audio_file}")
            continue

        # 检查是否已经处理过
        output_file = os.path.join(category_output_path, f"{os.path.splitext(audio_file)[0]}.png")
        if os.path.exists(output_file):
            print(f"已处理音频，跳过：{audio_file}")
            continue

        print(f"正在处理音频文件：{audio_file}")

        try:
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=None)  # sr=None 保持原采样率

            # 计算 MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc = np.clip(mfcc, a_min=-100, a_max=100)  # 限制动态范围，避免异常值

            # 创建频谱图图像
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, sr=sr, x_axis='time', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'MFCC - {audio_file}')  # 支持中文的标题
            plt.tight_layout()

            # 保存频谱图为图像
            plt.savefig(output_file)
            plt.close()

            print(f"频谱图已保存至：{output_file}")
        except Exception as e:
            print(f"处理音频文件失败：{audio_file}, 错误信息：{e}")
            failed_audio_count += 1  # 增加失败计数
            failed_audio_list.append(f"{category}/{audio_file}")  # 记录失败文件路径

# 打印失败统计结果
print("\n音频文件处理完成！")
if failed_audio_count > 0:
    print(f"总共有 {failed_audio_count} 个音频文件处理失败：")
    for failed_audio in failed_audio_list:
        print(f" - {failed_audio}")
else:
    print("所有音频文件均处理成功！")
