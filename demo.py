#单独测试模型对一个视频的预测效果

from process import process_single_video
from extra_frame_feature import extract_single_video_feature
from extra_audio_feature import extract_audio_feature_for_video, define_vggish_model
from extra_title_feature import extract_title_feature_for_video
from nextvlad_cluster import aggregate_single_video_features
import os
import numpy as np
import pandas as pd
from train import MultiModalWeightedFusion
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
import matplotlib.pyplot as plt
from PIL import Image
import os

CATEGORIES = [
    "车辆文化", "动物", "二创", "仿妆cos", "搞笑", "鬼畜", "绘画", "科技", "科普", "美食",
    "明星综合", "人文历史", "三农", "设计·创意", "生活", "时尚潮流", "手办·模玩", "特摄",
    "舞蹈", "音乐", "影视", "游戏", "运动综合", "职业综合", "综合"
]

# 输入视频文件路径
video_path = r"E:/毕业设计/datasets/bili_datasets/raw_data/test/video/绘画/三个星期实现了《锦鲤玉扇》的开扇动画，第一次画会动的水。 [BV19Y4y1E7zE].mp4"
output_base_dir = r"E:/毕业设计/bishe_test"
output_dir = r"E:/毕业设计/bishe_test/features/frame_features"

# 设置音频文件所在的目录
audio_base_path = r"E:/毕业设计/bishe_test/audio"
audio_features_folder = r"E:/毕业设计/bishe_test/features/audio_features"
os.makedirs(audio_features_folder, exist_ok=True)  # 创建输出文件夹

def save_audio_feature_to_csv(audio_feature, cleaned_title):
    # 保存音频特征向量到 CSV 文件
    if audio_feature is not None:
        feature_file_path = os.path.join(audio_features_folder, f"{cleaned_title}.csv")

        # 创建列名：file_name + feature_0, feature_1, ...
        column_names = ["file_name"] + [f"feature_{i}" for i in range(audio_feature.shape[1])]

        # 为每一行特征加上文件名
        data_with_filename = [[cleaned_title] + vector.tolist() for vector in audio_feature]

        # 创建 DataFrame
        df = pd.DataFrame(data_with_filename, columns=column_names)

        # 保存为 CSV 文件
        df.to_csv(feature_file_path, index=False, encoding="utf-8-sig")
        print(f"音频特征已保存至：{feature_file_path}")
    else:
        print(f"音频特征为空，无法保存：{cleaned_title}")

def save_title_feature_to_csv(title_feature, cleaned_title):
    """
    保存标题特征向量到 CSV 文件，路径为：
    E:/毕业设计/bishe_test/features/title_features/{cleaned_title}.csv
    """
    if title_feature is not None:
        title_features_folder = r"E:/毕业设计/bishe_test/features/title_features"
        os.makedirs(title_features_folder, exist_ok=True)

        feature_file_path = os.path.join(title_features_folder, f"{cleaned_title}.csv")

        # 将特征转为 DataFrame，第一列为 file_name
        df = pd.DataFrame([[cleaned_title] + title_feature.tolist()],
                          columns=["file_name"] + [f"feature_{i}" for i in range(len(title_feature))])

        df.to_csv(feature_file_path, index=False, encoding="utf-8-sig")
        print(f"标题特征已保存至：{feature_file_path}")
    else:
        print(f"标题特征为空，无法保存：{cleaned_title}")

def load_feature_vector(path):
    df = pd.read_csv(path)
    vector = df.iloc[0, 1:].to_numpy(dtype=np.float32)
    return torch.tensor(vector)

def show_prediction_window(cleaned_title, raw_title, top5_indices, top5_probs, sort_txt_path, frame_dir):
    # 1. 获取第一帧图像
    first_frame_path = None
    for fname in sorted(os.listdir(frame_dir)):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            first_frame_path = os.path.join(frame_dir, fname)
            break

    if not first_frame_path:
        print("未找到第一帧图像")
        return

    # 2. 加载图片
    image = Image.open(first_frame_path)

    # 3. 加载源标签（从 sort.txt 中匹配 raw_title）
    original_label = "未知"
    with open(sort_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if raw_title in line:
                original_label = line.strip().split('\t')[-1]
                break

    # 4. 获取预测标签名及概率
    predicted_lines = []
    for idx, prob in zip(top5_indices, top5_probs):
        category = CATEGORIES[idx]
        predicted_lines.append(f"{category} ({prob:.2%})")

    # 5. 展示图像 + 标签信息
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')

    # 设置标题显示原始和预测标签
    plt.title(f"原始标签：{original_label}", fontsize=12, color='blue', loc='left')

    # 添加预测文本信息
    prediction_text = "\n".join([f"Top {i+1}: {line}" for i, line in enumerate(predicted_lines)])
    plt.figtext(0.5, 0.01, prediction_text, wrap=True, ha='center', fontsize=14, color='white', backgroundcolor='black')

    plt.tight_layout()
    plt.show()
    
def predict(cleaned_title, raw_title, model_path):
    # 特征路径
    image_feat_path = rf"E:\毕业设计\bishe_test\features\feature_nextvlad\{cleaned_title}_frame_nextvlad.csv"
    audio_feat_path = rf"E:\毕业设计\bishe_test\features\feature_nextvlad\{cleaned_title}_audio_nextvlad.csv"
    text_feat_path = rf"E:\毕业设计\bishe_test\features\title_features\{cleaned_title}.csv"

    # 加载特征向量
    image_feat = load_feature_vector(image_feat_path).unsqueeze(0)  # (1, 2048)
    audio_feat = load_feature_vector(audio_feat_path).unsqueeze(0)  # (1, 512)
    text_feat = load_feature_vector(text_feat_path).unsqueeze(0)    # (1, 768)

    # 加载模型
    model = MultiModalWeightedFusion(use_attention=True)  #V6模型选False,V7模型选True
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        logits, weights = model(image_feat, audio_feat, text_feat)
        probs = F.softmax(logits, dim=1).squeeze().numpy()
        top5_idx = probs.argsort()[-5:][::-1]
        print("Top-5 预测结果:")
        for i in top5_idx:
            print(f"类别: {CATEGORIES[i]} - 概率: {probs[i]:.4f}")
        print("模态融合权重:", F.softmax(weights, dim=0).detach().numpy())
        
         # 添加展示窗口
        frame_dir = os.path.join(output_base_dir, "frames", cleaned_title)
        sort_txt_path = r"E:/毕业设计/datasets/bili_datasets/raw_data/sort.txt"
        show_prediction_window(cleaned_title,raw_title, top5_idx, probs[top5_idx], sort_txt_path, frame_dir)
        
def main():
    sess, embeddings, graph = define_vggish_model()
    
    # 提取指定视频的音频、帧图像、标题
    frame_dir, audio_file, cleaned_title = process_single_video(video_path, output_base_dir)
    
    if not frame_dir or not audio_file:
        print("视频预处理失败")
        return None

    # 通过 cleaned_title 生成音频文件路径
    audio_filename = f"{cleaned_title}.wav"
    audio_path = os.path.join(audio_base_path, audio_filename)

    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"找不到音频文件：{audio_path}")
        return

    # 提取视频帧特征
    print("正在提取帧图像特征")
    frame_feature = extract_single_video_feature(frame_dir, output_dir, cleaned_title)
    
    # 提取音频帧特征
    print(f"正在提取音频特征：{audio_filename}")
    # 传递完整的 audio_path 和已加载的 sess, graph 和 embeddings 给 extract_audio_feature_for_video 函数
    audio_feature = extract_audio_feature_for_video(audio_path, sess, graph, embeddings)
    
    # 提取标题特征
    print(f"正在提取标题特征")
    title_feature = extract_title_feature_for_video(cleaned_title)
    
    # 保存音频特征到 CSV 文件
    save_audio_feature_to_csv(audio_feature, cleaned_title)
    save_title_feature_to_csv(title_feature, cleaned_title)
    
    print(f"frame_feature type: {type(frame_feature)}")
    print(f"audio_feature type: {type(audio_feature)}")
    print(f"title_feature type: {type(title_feature)}")


    #聚合指定视频的帧图像特征和音频帧特征
    #print(f"正在进行图像和音频的特征聚合")
    aggregate_single_video_features(cleaned_title)
    
    print("视频处理完成")
    print(f"帧图像目录: {frame_dir}")
    print(f"音频文件路径: {audio_file}")
    print(f"清理后的标题: {cleaned_title}")
    
    # 从视频路径中提取原始标题
    raw_filename = os.path.splitext(os.path.basename(video_path))[0]
    raw_title = raw_filename.rsplit('[', 1)[0].strip()

    
    return cleaned_title, raw_title

if __name__ == "__main__":
    cleaned_title, raw_title = main()
    if cleaned_title is not None:
        model_path = r"E:/毕业设计/src/checkpoints/V7_best_model.pt"
        predict(cleaned_title,raw_title, model_path)
    else:
        print("清理后的标题为空，预测终止。")