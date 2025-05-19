# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Global parameters for the VGGish model.

See vggish_slim.py for more information.
"""

# Architectural constants.
NUM_FRAMES = 96  # 输入 Mel 频谱图的帧数
NUM_BANDS = 64  # 输入 Mel 频谱图的频率带数
EMBEDDING_SIZE = 128  # 嵌入层的大小

# 用于特征和样本生成的超参数
SAMPLE_RATE = 16000 # 采样率（Hz）
STFT_WINDOW_LENGTH_SECONDS = 0.025 # STFT（短时傅立叶变换）窗口长度（秒）
STFT_HOP_LENGTH_SECONDS = 0.010 # STFT 滑动步长（秒）
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # 用于稳定化对数计算的偏移量
EXAMPLE_WINDOW_SECONDS = 0.96  # 每个样本包含 96 帧，每帧 10 毫秒
EXAMPLE_HOP_SECONDS = 0.96     # 样本之间无重叠

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
PCA_MEANS_NAME = 'pca_means'
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = 'vggish/input_features'
INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
OUTPUT_OP_NAME = 'vggish/embedding'
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'
