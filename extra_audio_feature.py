import os
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
import soundfile as sf
from vggish import vggish_input, vggish_params, vggish_slim
import pandas as pd

# 文件路径
audio_base_path = r"E:/毕业设计/datasets/bili_datasets/raw_data/test/audio"
output_base_path = r"E:/毕业设计/datasets/bili_datasets/features/audio_vggish/test"

# 创建输出目录
os.makedirs(output_base_path, exist_ok=True)

# VGGish 模型的检查点路径
checkpoint_path = 'E:/毕业设计/src/vggish/vggish_model.ckpt'

# 定义 VGGish 模型
def define_vggish_model():
    graph = tf.Graph()  # 创建新的图
    with graph.as_default():
        sess = tf.Session()
        embeddings = vggish_slim.define_vggish_slim(training=False)  # VGGish 作为特征提取器
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)  # 加载预训练权重
    return sess, embeddings, graph

# 加载 VGGish 模型
sess, embeddings ,graph= define_vggish_model()

if __name__ == "__main__":
    # 遍历音频分类目录
    categories = os.listdir(audio_base_path)

    for category in categories:
        category_path = os.path.join(audio_base_path, category)
        if not os.path.isdir(category_path):
            continue

        # 输出文件路径
        category_output_file = os.path.join(output_base_path, f"{category}.csv")
        all_features = []  # 存储该分类的新增特征向量

        # 已处理音频列表
        processed_files = set()
        if os.path.exists(category_output_file):
            try:
                df_existing = pd.read_csv(category_output_file, encoding="utf-8-sig")
                processed_files = set(df_existing["文件名"].tolist())
            except Exception as e:
                print(f"读取已有特征文件失败：{category_output_file}，错误信息：{e}")

        # 遍历音频文件
        audio_files = os.listdir(category_path)
        for audio_file in audio_files:
            if not audio_file.endswith(".wav"):
                print(f"跳过非 WAV 文件：{audio_file}")
                continue

            if audio_file in processed_files:
                print(f"跳过已处理音频文件：{audio_file}")
                continue

            print(f"正在处理音频文件：{audio_file}")
            audio_path = os.path.join(category_path, audio_file)

            try:
                # 读取音频文件
                waveform, sr = sf.read(audio_path)

                # 转换为 log-Mel 频谱图
                log_mel = vggish_input.waveform_to_examples(waveform, sr)

                # 提取 VGGish 嵌入特征
                features_input = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                [feature_vector] = sess.run([embeddings], feed_dict={features_input: log_mel})

                # 保存特征向量到列表
                for vector in feature_vector:
                    all_features.append([audio_file] + vector.tolist())  # 文件名 + 特征值
            except Exception as e:
                print(f"处理音频文件失败：{audio_file}, 错误信息：{e}")

        # 追加写入新特征
        if all_features:
            df_new = pd.DataFrame(all_features)
            df_new.columns = ["文件名"] + [f"特征{i}" for i in range(df_new.shape[1] - 1)]
            if os.path.exists(category_output_file):
                df_new.to_csv(category_output_file, mode='a', index=False, encoding="utf-8-sig", header=False)
            else:
                df_new.to_csv(category_output_file, index=False, encoding="utf-8-sig")
            print(f"{category} 分类新增特征向量已保存至：{category_output_file}")
        else:
            print(f"{category} 分类没有需要新增处理的音频文件。")


def extract_audio_feature_for_video(audio_filename, sess, graph, embeddings):
    audio_path = audio_filename  # 直接使用传入的音频文件路径

    if not audio_filename.endswith(".wav"):
        print(f"跳过非 WAV 文件：{audio_filename}")
        return None

    try:
        # 读取音频文件
        waveform, sr = sf.read(audio_path)
        
        # 转换为 log-Mel 频谱图
        log_mel = vggish_input.waveform_to_examples(waveform, sr)

        # 需要明确指定计算图
        with graph.as_default():
            features_input = graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            [feature_vector] = sess.run([embeddings], feed_dict={features_input: log_mel})

        return feature_vector  # 返回特征向量 [N, 128]
    
    except Exception as e:
        print(f"处理音频失败：{audio_filename}, 错误信息：{e}")
        return None
