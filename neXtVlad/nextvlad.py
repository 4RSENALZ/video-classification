from .frame_level_models import *
from absl import flags
from tensorflow.keras import layers as tf_layers
import tf_slim as slim
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_integer("nextvlad_cluster_size", 64, "Number of units in the NeXtVLAD cluster layer.")
flags.DEFINE_integer("nextvlad_hidden_size", 1024, "Number of units in the NeXtVLAD hidden layer.")

flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
flags.DEFINE_float("drop_rate", 0.5, "dropout ratio after VLAD encoding")
flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
flags.DEFINE_integer("gating_reduction", 8, "reduction factor in se context gating")

flags.DEFINE_integer("mix_number", 3, "the number of gvlad models")
flags.DEFINE_float("cl_temperature", 2, "temperature in collaborative learning")
flags.DEFINE_float("cl_lambda", 1.0, "penalty factor of cl loss")


class NeXtVLAD(tf.keras.layers.Layer):
    def __init__(self, feature_size, max_frames, cluster_size, is_training=True, expansion=2, groups=None, **kwargs):
        super(NeXtVLAD, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

        # Initialize cluster weights and batch normalization layers
        self.expansion_fc = tf.keras.layers.Dense(
            expansion * feature_size, activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        self.attention_fc = tf.keras.layers.Dense(
            groups, activation="sigmoid",
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        self.cluster_weights = self.add_weight(
            name="cluster_weights",
            shape=(self.expansion * feature_size, self.groups * self.cluster_size),
            initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            trainable=True
        )
        self.cluster_weights2 = self.add_weight(
            name="cluster_weights2",
            shape=(1, (self.expansion * feature_size) // self.groups, self.cluster_size),
            initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            trainable=True
        )
        self.bn_cluster = tf.keras.layers.BatchNormalization()
        self.bn_vlad = tf.keras.layers.BatchNormalization()

    def call(self, inputs, mask=None):
        # Fully connected for expansion
        inputs_expanded = self.expansion_fc(inputs)# 通过全连接层扩展特征维度，使其更适合 VLAD 聚类

        # Attention mechanism
        attention = self.attention_fc(inputs_expanded)#计算每一帧的注意力权重
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, axis=-1))
        attention = tf.reshape(attention, [-1, self.max_frames * self.groups, 1])    # 展平时间维度和 group 维度，准备参与后续矩阵运算

        # Compute activation
        inputs_reshaped = tf.reshape(inputs_expanded, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(inputs_reshaped, self.cluster_weights)    # 计算输入特征与聚类中心的相似度（Soft Assignment软分配）
        activation = self.bn_cluster(activation, training=self.is_training)
        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)# 使用 softmax 得到软分配概率（每帧特征属于每个 cluster 的概率）

        # Apply attention
        # 用 attention 权重对 soft-assignment 进行加权，强调重要帧的贡献
        activation = tf.multiply(activation, attention)
        # 计算每个 cluster 的残差总和需要一个均值项 a，a_sum 是所有帧对每个 cluster 的总贡献权重
        a_sum = tf.reduce_sum(activation, axis=-2, keepdims=True)
        a = tf.multiply(a_sum, self.cluster_weights2)

        # Calculate VLAD
        activation = tf.transpose(activation, perm=[0, 2, 1])
        inputs_grouped = tf.reshape(inputs_expanded, [-1, self.max_frames * self.groups,
                                                      (self.expansion * self.feature_size) // self.groups])
        vlad = tf.matmul(activation, inputs_grouped)# 执行加权求和：对每个 cluster 加权聚合其所“负责”的帧特征
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)    # 减去 cluster 中心均值，得到残差向量：表示特征与 cluster 中心的偏差

        # Normalization and final transformation
        # 对每个通道做 L2 归一化，防止某些 cluster 的输出过大
        vlad = tf.nn.l2_normalize(vlad, axis=1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * ((self.expansion * self.feature_size) // self.groups)])    # 将所有 cluster 的残差拼接成一个向量，形成最终聚合特征
        vlad = self.bn_vlad(vlad, training=self.is_training)

        return vlad
#call 方法会返回聚合后的特征，形状为 (1, cluster_size * ((expansion * feature_size) // groups))
#例如我的输入特征维度是 512，扩展倍数是 2，分组数是 8，聚类数是 16，那么输出特征的维度就是 (1, 16 * ((2 * 512) // 8)) = (1, 2048)

class NeXtVLADModel_teacher(models.BaseModel):
    """Creates a NeXtVLAD based model.
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

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     groups=None,
                     expansion=None,
                     drop_rate=None,
                     gating_reduction=None,
                     k_frame=300,
                     **unused_params):
        cluster_size = cluster_size or FLAGS.nextvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.nextvlad_hidden_size
        gating_reduction = gating_reduction or FLAGS.gating_reduction
        groups = groups or FLAGS.groups
        drop_rate = drop_rate or FLAGS.drop_rate
        expansion = expansion or FLAGS.expansion

        #teacher model不需要切片。直接用上所有帧就对了

        mask = tf.sequence_mask(num_frames, 300, dtype=tf.float32)
        #一个二维mask数组 ，参考https://blog.csdn.net/wxf2012301351/article/details/84401950
        #80行，300列，每行是一个video，前num_frames[i]个元素是1，后面是0

        max_frames = model_input.get_shape().as_list()[1] # max_frames=300，已经是固定的了呀。。。     get_shape是tensor专用，返回的是元组，再用as_list()转成list

        video_nextvlad = NeXtVLAD(1024, max_frames, cluster_size, is_training, groups=groups, expansion=expansion)
        audio_nextvlad = NeXtVLAD(128, max_frames, cluster_size // 2, is_training, groups=groups // 2, expansion=expansion)

        print('teacher_model_input_forward=',model_input)
        with tf.variable_scope("video_VLAD"):
            vlad_video = video_nextvlad(model_input[:, :, 0:1024], mask=mask) #就是在这里前向传播的！？

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_nextvlad(model_input[:, :, 1024:], mask=mask)

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        if drop_rate > 0.:
            vlad = slim.dropout(vlad, keep_prob=1. - drop_rate, is_training=is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        # activation = tf.nn.relu(activation)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden1_size, hidden1_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden1_size // gating_reduction, hidden1_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)
        tf.summary.histogram("final_gates", gates)

        activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model) #就是LogisticModel

        return activation,aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)





class NeXtVLADModel_student(models.BaseModel):
    """Creates a NeXtVLAD based model.
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

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     groups=None,
                     expansion=None,
                     drop_rate=None,
                     gating_reduction=None,
                     k_frame=300,
                     **unused_params):
        cluster_size = cluster_size or FLAGS.nextvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.nextvlad_hidden_size
        gating_reduction = gating_reduction or FLAGS.gating_reduction
        groups = groups or FLAGS.groups
        drop_rate = drop_rate or FLAGS.drop_rate
        expansion = expansion or FLAGS.expansion

        #student model需要对输入的mask操作进行切片
        frame_step=int(300/k_frame)
        mask = tf.sequence_mask(num_frames, 300, dtype=tf.float32)
        print('mask=', mask)
        mask=mask[ : , 0:300:frame_step ]
        print('mask_student=', mask)
        #一个二维mask数组 ，参考https://blog.csdn.net/wxf2012301351/article/details/84401950
        #80行，300列，每行是一个video，前num_frames[i]个元素是1，后面是0

        max_frames = model_input.get_shape().as_list()[1] # max_frames=300，已经是固定的了呀。。。     get_shape是tensor专用，返回的是元组，再用as_list()转成list
        print('max_frames_student=', max_frames)
        video_nextvlad = NeXtVLAD(1024, max_frames, cluster_size, is_training, groups=groups, expansion=expansion)
        audio_nextvlad = NeXtVLAD(128, max_frames, cluster_size // 2, is_training, groups=groups // 2, expansion=expansion)

        print('model_input_forward=',model_input)
        with tf.variable_scope("video_VLAD"):
            vlad_video = video_nextvlad(model_input[:, :, 0:1024], mask=mask) #就是在这里前向传播的！？

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_nextvlad(model_input[:, :, 1024:], mask=mask)

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        if drop_rate > 0.:
            vlad = slim.dropout(vlad, keep_prob=1. - drop_rate, is_training=is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        # activation = tf.nn.relu(activation)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden1_size, hidden1_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden1_size // gating_reduction, hidden1_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)
        tf.summary.histogram("final_gates", gates)

        activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,FLAGS.video_level_classifier_model) #就是LogisticModel

        return activation,aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)























class MixNeXtVladModel(models.BaseModel):

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     mix_number=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     groups=None,
                     expansion=None,
                     drop_rate=None,
                     gating_reduction=None,
                     **unused_params):
        cluster_size = cluster_size or FLAGS.nextvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.nextvlad_hidden_size
        gating_reduction = gating_reduction or FLAGS.gating_reduction
        groups = groups or FLAGS.groups
        drop_rate = drop_rate or FLAGS.drop_rate
        mix_number = mix_number or FLAGS.mix_number
        expansion = expansion or FLAGS.expansion
        mask = tf.sequence_mask(num_frames, 300, dtype=tf.float32)

        max_frames = model_input.get_shape().as_list()[1]

        ftr_mean = tf.reduce_mean(model_input, axis=-1)
        ftr_mean = slim.batch_norm(ftr_mean,
                                   center=True,
                                   scale=True,
                                   fused=True,
                                   is_training=is_training,
                                   scope="mix_weights_bn")
        mix_weights = slim.fully_connected(ftr_mean, mix_number, activation_fn=None,
                                           weights_initializer=slim.variance_scaling_initializer(),
                                           scope="mix_weights")
        mix_weights = tf.nn.softmax(mix_weights, axis=-1)
        tf.summary.histogram("mix_weights", mix_weights)

        results = []
        for n in range(mix_number):
            with tf.variable_scope("branch_%d"%n):
                res = self.nextvlad_model(video_ftr=model_input[:, :, 0:1024], audio_ftr=model_input[:, :, 1024:], vocab_size=vocab_size,
                                          max_frames=max_frames, cluster_size=cluster_size, groups=groups, expansion=expansion,
                                          drop_rate=drop_rate, hidden1_size=hidden1_size, is_training=is_training,
                                          gating_reduction=gating_reduction, mask=mask, **unused_params)
                results.append(res)

        aux_preds = [res["predictions"] for res in results]
        logits = [res["logits"] for res in results]
        logits = tf.stack(logits, axis=1)

        mix_logit = tf.reduce_sum(tf.multiply(tf.expand_dims(mix_weights, -1), logits), axis=1)

        pred = tf.nn.sigmoid(mix_logit)

        if is_training:
            rank_pred = tf.expand_dims(tf.nn.softmax(tf.div(mix_logit, FLAGS.cl_temperature), axis=-1), axis=1)
            aux_rank_preds = tf.nn.softmax(tf.div(logits, FLAGS.cl_temperature), axis=-1)
            epsilon = 1e-8
            kl_loss = tf.reduce_sum(rank_pred * (tf.log(rank_pred + epsilon) - tf.log(aux_rank_preds + epsilon)),
                                    axis=-1)

            regularization_loss = FLAGS.cl_lambda * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1), axis=-1)

            return  {"predictions": pred,
                     "regularization_loss": regularization_loss,
                     "aux_predictions": aux_preds}
        else:
            return {"predictions": pred}
            # return {"predictions": results[0]["predictions"]}

    def nextvlad_model(self, video_ftr, audio_ftr, vocab_size, max_frames,
                       cluster_size, groups, drop_rate, hidden1_size,
                       is_training, gating_reduction, mask, expansion,
                       **unused_params):
        video_vlad = NeXtVLAD(1024, max_frames, cluster_size=cluster_size, groups=groups, expansion=expansion,
                              is_training=is_training)
        audio_vlad = NeXtVLAD(128, max_frames, cluster_size=cluster_size // 2, groups=groups // 2, expansion=expansion,
                              is_training=is_training)

        with tf.variable_scope("video_vlad"):
            video_ftr = video_vlad(video_ftr, mask=mask)
        with tf.variable_scope("audio_vlad"):
            audio_ftr = audio_vlad(audio_ftr, mask=mask)

        vlad = tf.concat([video_ftr, audio_ftr], 1)

        if drop_rate > 0.:
            vlad = slim.dropout(vlad, keep_prob=1. - drop_rate, is_training=is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden1_size, hidden1_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden1_size // gating_reduction, hidden1_size],
                                           initializer=slim.variance_scaling_initializer())
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)