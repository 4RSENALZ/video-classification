import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# 目录路径
frames_base_path = r"E:/毕业设计/datasets/bili_datasets/processed/frames/test"
features_base_path = r"E:/毕业设计/datasets/bili_datasets/features/text_bert/test"

# 创建输出目录（如果不存在）
os.makedirs(features_base_path, exist_ok=True)

# 加载 BERT 预训练模型和分词器
bert_model_path = r"E:/毕业设计/src/bert_base_chinese"
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model = AutoModel.from_pretrained(bert_model_path)

model.eval()  # 设为评估模式

if __name__ == "__main__":
    # 遍历分类目录
    categories = os.listdir(frames_base_path)

    for category in categories:
        category_path = os.path.join(frames_base_path, category)
        if not os.path.isdir(category_path):
            continue  # 不是目录就跳过

        # 创建对应的分类特征目录
        category_output_path = os.path.join(features_base_path, category)
        os.makedirs(category_output_path, exist_ok=True)

        # 加载已经存在的特征文件（如果存在）
        category_output_file = os.path.join(category_output_path, f"{category}.csv")
        processed_titles = set()
        if os.path.exists(category_output_file):
            existing_df = pd.read_csv(category_output_file, encoding="utf-8-sig")
            processed_titles = set(existing_df["file_name"].tolist())  # 已经处理过的视频标题集合

        all_features = []  # 存储所有新提取的视频标题特征向量

        # 遍历该分类下的视频文件夹（文件夹名就是视频标题）
        video_folders = os.listdir(category_path)
        for video_title in video_folders:
            video_path = os.path.join(category_path, video_title)
            if not os.path.isdir(video_path):
                continue  # 不是文件夹就跳过

            # 检查是否已处理过
            if video_title in processed_titles:
                print(f"已处理视频标题，跳过：{video_title}")
                continue

            print(f"正在处理视频标题：{video_title}")

            try:
                # 使用 BERT 分词器进行编码
                inputs = tokenizer(video_title, return_tensors="pt", padding=True, truncation=True, max_length=32)

                # 提取 BERT 特征
                with torch.no_grad():
                    outputs = model(**inputs)
                    feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  #在 BERT中输入序列的第一个 token 始终是 [CLS]，[:, 0, :]就是选取 [CLS] token作为句子整体的语义表示

                # 存储特征
                all_features.append([video_title] + feature_vector.tolist())

            except Exception as e:
                print(f"处理视频标题失败：{video_title}, 错误信息：{e}")

        # 将新特征追加保存到 CSV 文件
        if all_features:
            new_df = pd.DataFrame(all_features, columns=["file_name"] + [f"feature{i}" for i in range(len(all_features[0]) - 1)])
            if os.path.exists(category_output_file):
                existing_df = pd.read_csv(category_output_file, encoding="utf-8-sig")
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            final_df.to_csv(category_output_file, index=False, encoding="utf-8-sig")
            print(f"{category} 分类的 BERT 特征已保存至：{category_output_file}")
        else:
            print(f"{category} 分类没有新的 BERT 特征需要保存，跳过。")

def extract_title_feature_for_video(title: str):
    """提取单个视频标题的 BERT 特征向量（返回 numpy 向量）"""
    try:
        inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = model(**inputs)
            feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
        return feature_vector
    except Exception as e:
        print(f"提取标题特征失败：{title}, 错误信息：{e}")
        return None