"""Contains a collection of models which operate on variable-length sequences.
"""
import math
import tensorflow as tf
from neXtVlad import models
from neXtVlad import video_level_models
from neXtVlad import model_utils as utils

#对视频帧平均池化，用逻辑回归分类器对池化结果进行分类
class FrameLevelLogisticModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a logistic classifier over the average of the
        frame-level features.

        Args:
            model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                         input features.
            vocab_size: The number of classes in the dataset.
            num_frames: A vector of length 'batch' which indicates the number of
                 frames for each video (before padding).

        Returns:
            A dictionary with a tensor containing the probability predictions of the
            model in the 'predictions' key. The dimensions of the tensor are
            'batch_size' x 'num_classes'.
        """
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        feature_size = model_input.shape[2]

        denominators = tf.reshape(
            tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
        avg_pooled = tf.reduce_sum(model_input, axis=1) / denominators

        output = tf.keras.layers.Dense(
            vocab_size, activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.L2(1e-8))(avg_pooled)
        return {"predictions": output}

class DbofModel(models.BaseModel):
    """Creates a Deep Bag of Frames model."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=30,
                     add_batch_norm=True,
                     sample_random_frames=True,
                     cluster_size=8192,
                     hidden_size=1024,
                     is_training=True,
                     **unused_params):
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if sample_random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames, iterations)

        max_frames = model_input.shape[1]
        feature_size = model_input.shape[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        if add_batch_norm:
            reshaped_input = tf.keras.layers.BatchNormalization()(reshaped_input)

        cluster_weights = tf.Variable(
            initial_value=tf.random.normal([feature_size, cluster_size],
                                           stddev=1 / math.sqrt(feature_size)),
            name="cluster_weights")
        activation = tf.matmul(reshaped_input, cluster_weights)

        if add_batch_norm:
            activation = tf.keras.layers.BatchNormalization()(activation)
        else:
            cluster_biases = tf.Variable(
                initial_value=tf.random.normal([cluster_size],
                                               stddev=1 / math.sqrt(feature_size)),
                name="cluster_biases")
            activation += cluster_biases

        activation = tf.nn.relu6(activation)
        activation = tf.reshape(activation, [-1, max_frames, cluster_size])
        activation = utils.FramePooling(activation, "max")

        hidden1_weights = tf.Variable(
            initial_value=tf.random.normal([cluster_size, hidden_size],
                                           stddev=1 / math.sqrt(cluster_size)),
            name="hidden1_weights")
        activation = tf.matmul(activation, hidden1_weights)

        if add_batch_norm:
            activation = tf.keras.layers.BatchNormalization()(activation)
        else:
            hidden1_biases = tf.Variable(
                initial_value=tf.random.normal([hidden_size], stddev=0.01),
                name="hidden1_biases")
            activation += hidden1_biases

        activation = tf.nn.relu6(activation)

        aggregated_model = getattr(video_level_models, "MoeModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            **unused_params)

class LstmModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.

        Args:
            model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                         input features.
            vocab_size: The number of classes in the dataset.
            num_frames: A vector of length 'batch' which indicates the number of
                 frames for each video (before padding).

        Returns:
            A dictionary with a tensor containing the probability predictions of the
            model in the 'predictions' key. The dimensions of the tensor are
            'batch_size' x 'num_classes'.
        """
        lstm_size = 1024
        number_of_layers = 2

        lstm_cells = [tf.keras.layers.LSTMCell(lstm_size) for _ in range(number_of_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)

        outputs, state = tf.keras.layers.RNN(stacked_lstm, return_state=True)(
            model_input, mask=tf.sequence_mask(num_frames))

        aggregated_model = getattr(video_level_models, "MoeModel")
        return aggregated_model().create_model(
            model_input=state[-1],
            vocab_size=vocab_size,
            **unused_params)
