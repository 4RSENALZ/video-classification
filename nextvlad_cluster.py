# 把分帧后的图像特征向量聚合
# 以及把分帧后的音频特征向量聚合

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from neXtVlad import nextvlad
from neXtVlad.nextvlad import NeXtVLAD

# 分类标签
categories = [
    "车辆文化", "动物", "二创", "仿妆cos", "搞笑", "鬼畜", "绘画", "科技", "科普", "美食",
    "明星综合", "人文历史", "三农", "设计·创意", "生活", "时尚潮流", "手办·模玩", "特摄",
    "舞蹈", "音乐", "影视", "游戏", "运动综合", "职业综合", "综合"
]

# NeXtVLAD 参数
cluster_size = 16
groups = 8
dropout_rate = 0.2
max_frames = 300

# 构建 NeXtVLAD 模型
def build_nextvlad_model(feature_dim):
    inputs = tf.keras.Input(shape=(None, feature_dim), name="frame_features")
    nextvlad_layer = NeXtVLAD(
        feature_size=feature_dim,
        max_frames=max_frames,
        cluster_size=cluster_size,
        is_training=False,
        expansion=2,
        groups=groups
    )
    outputs = nextvlad_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# 聚合函数
def aggregate_features_with_nextvlad(model, feature_array):
    num_frames = feature_array.shape[0]

    if num_frames > max_frames:
        feature_array = feature_array[:max_frames, :]
    elif num_frames < max_frames:
        padding = np.zeros((max_frames - num_frames, feature_array.shape[1]), dtype=np.float32)
        feature_array = np.vstack((feature_array, padding))

    feature_array = np.expand_dims(feature_array, axis=0)
    aggregated = model.predict(feature_array, verbose=0)#predict 方法会将输入数据传递给模型的前向传播逻辑（即 call 方法）
    return aggregated.squeeze()

if __name__ == "__main__":
    # ========== 批量聚合音频分类特征 ==========
    print("开始处理音频特征聚合...")
    audio_input_dir = r"E:/毕业设计/datasets/bili_datasets/features/audio_vggish/test"
    audio_output_dir = r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/test/audio_nextvlad"
    os.makedirs(audio_output_dir, exist_ok=True)
    audio_model = build_nextvlad_model(feature_dim=128)

    for category in categories:
        csv_path = os.path.join(audio_input_dir, f"{category}.csv")
        if not os.path.exists(csv_path):
            print(f"未找到分类：{category} 的音频特征文件，跳过。")
            continue

        print(f"\n处理音频分类：{category}")
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"{category} 音频特征为空，跳过。")
            continue

        df.columns = ["file_name"] + [f"feature_{i}" for i in range(1, df.shape[1])]
        aggregated_rows = []

        grouped = df.groupby("file_name")
        for file_name, group in grouped:
            features = group.drop(columns=["file_name"]).to_numpy(dtype=np.float32)
            try:
                agg_feature = aggregate_features_with_nextvlad(audio_model, features)
                aggregated_rows.append([file_name] + agg_feature.tolist())
            except Exception as e:
                print(f"聚合失败：{file_name}，错误：{e}")

        if aggregated_rows:
            output_df = pd.DataFrame(aggregated_rows)
            output_file = os.path.join(audio_output_dir, f"{category}_nextvlad.csv")
            output_df.to_csv(output_file, index=False, encoding="utf-8-sig",
                             header=["file_name"] + [f"agg_feat_{i}" for i in range(audio_model.output_shape[1])])
            print(f"已保存音频聚合特征：{output_file}")
        else:
            print(f"{category} 无有效音频特征，未保存。")

    # ========== 图像聚合 ==========
    print("\n开始处理图像特征聚合...")
    image_input_root = r"E:/毕业设计/datasets/bili_datasets/features/frame_resnet/test"
    image_output_dir = r"E:/毕业设计/datasets/bili_datasets/features/feature_nextvlad/test/frame_nextvlad"
    os.makedirs(image_output_dir, exist_ok=True)
    image_model = build_nextvlad_model(feature_dim=512)

    for category in categories:
        category_dir = os.path.join(image_input_root, category)
        if not os.path.isdir(category_dir):
            print(f"未找到分类目录：{category}，跳过。")
            continue

        output_rows = []

        for filename in os.listdir(category_dir):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(category_dir, filename)
            video_name = os.path.splitext(filename)[0]

            try:
                df = pd.read_csv(file_path)
                if df.empty or df.shape[1] < 2:
                    print(f"{video_name} 的图像特征为空或格式错误，跳过。")
                    continue

                features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
                agg_feature = aggregate_features_with_nextvlad(image_model, features)
                output_rows.append([video_name] + agg_feature.tolist())

            except Exception as e:
                print(f"处理 {video_name} 时出错：{e}")

        if output_rows:
            output_df = pd.DataFrame(output_rows)
            output_file = os.path.join(image_output_dir, f"{category}_nextvlad.csv")
            output_df.to_csv(output_file, index=False, encoding="utf-8-sig",
                             header=["file_name"] + [f"agg_feat_{i}" for i in range(image_model.output_shape[1])])
            print(f"图像聚合特征已保存：{output_file}")
        else:
            print(f"{category} 无有效图像特征，未保存。")


#聚合指定视频的图像音频特征
def aggregate_single_video_features(cleaned_title):
    print(f"\n正在聚合视频特征: {cleaned_title}")

    frame_feat_path = os.path.join(r"E:/毕业设计/bishe_test/features/frame_features", f"{cleaned_title}.csv")
    audio_feat_path = os.path.join(r"E:/毕业设计/bishe_test/features/audio_features", f"{cleaned_title}.csv")

    output_dir = r"E:/毕业设计/bishe_test/features/feature_nextvlad"
    os.makedirs(output_dir, exist_ok=True)

    # 图像特征聚合
    frame_model = build_nextvlad_model(feature_dim=512)
    if os.path.exists(frame_feat_path):
        df = pd.read_csv(frame_feat_path)
        features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        frame_agg_feat = aggregate_features_with_nextvlad(frame_model, features)
        frame_df = pd.DataFrame([[cleaned_title] + frame_agg_feat.tolist()],
                                columns=["file_name"] + [f"agg_feat_{i}" for i in range(frame_model.output_shape[1])])
        frame_out_path = os.path.join(output_dir, f"{cleaned_title}_frame_nextvlad.csv")
        frame_df.to_csv(frame_out_path, index=False, encoding="utf-8-sig")
        print(f"图像聚合特征已保存：{frame_out_path}")
    else:
        print(f"找不到图像特征文件：{frame_feat_path}")

    # 音频特征聚合
    audio_model = build_nextvlad_model(feature_dim=128)
    if os.path.exists(audio_feat_path):
        df = pd.read_csv(audio_feat_path)
        features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        audio_agg_feat = aggregate_features_with_nextvlad(audio_model, features)
        audio_df = pd.DataFrame([[cleaned_title] + audio_agg_feat.tolist()],
                                columns=["file_name"] + [f"agg_feat_{i}" for i in range(audio_model.output_shape[1])])
        audio_out_path = os.path.join(output_dir, f"{cleaned_title}_audio_nextvlad.csv")
        audio_df.to_csv(audio_out_path, index=False, encoding="utf-8-sig")
        print(f"音频聚合特征已保存：{audio_out_path}")
    else:
        print(f"找不到音频特征文件：{audio_feat_path}")