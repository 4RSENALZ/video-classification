"""Contains model definitions."""
import math

from neXtVlad import models
import tensorflow as tf
from neXtVlad import utils

from absl import flags
import tf_slim as slim  # pip install tf_slim

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 4,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_float(
    "l2_penalty", 1e-8,
    "The l2 penalty of classifier weights and bias"
)

class LogisticModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, l2_penalty=None, **unused_params):
        """Creates a logistic model.

        Args:
            model_input: 'batch' x 'num_features' matrix of input features.
            vocab_size: The number of classes in the dataset.

        Returns:
            A dictionary with a tensor containing the probability predictions of the
            model in the 'predictions' key. The dimensions of the tensor are
            batch_size x num_classes.
        """
        l2_penalty = l2_penalty or FLAGS.l2_penalty
        logits = slim.fully_connected(
            model_input,
            vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            biases_regularizer=slim.l2_regularizer(l2_penalty),
            weights_initializer=slim.variance_scaling_initializer())
        output = tf.nn.sigmoid(logits)
        return {"predictions": output, "logits": logits}


class MoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=None,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.

        Args:
            model_input: 'batch_size' x 'num_features' matrix of input features.
            vocab_size: The number of classes in the dataset.
            num_mixtures: Number of mixtures (excluding dummy expert).
            l2_penalty: L2 regularization penalty.

        Returns:
            A dictionary with predictions.
        """
        l2_penalty = l2_penalty or FLAGS.l2_penalty
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations, [-1, num_mixtures + 1]))
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations, [-1, num_mixtures]))

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, axis=1)

        final_probabilities = tf.reshape(
            final_probabilities_by_class_and_batch, [-1, vocab_size])

        return {"predictions": final_probabilities}
