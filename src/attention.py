from __future__ import annotations

import inspect
import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    """
    Attention model included in the decoder class, which describes on which part of the image model should focus its attention
    when generating text.
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self._init_params = self.get_init_parameters(locals())

    def call(self, inputs):
        """
        Defines a forward pass of an attention model through the model
        :param inputs: Data to be forwarded through the attention
        :return: The result of passing data through the model
        """
        features, hidden = inputs
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_init_parameters(self, local_arguments):
        """
        Extracts arguments that are passed to the class initializer.
        :param local_arguments: Dictionary with all local arguments from the class initializer
        :return: Dictionary with all required arguments from the class initializer
        """
        init_params_dict = dict(inspect.signature(self.__init__).parameters)
        for key in init_params_dict.keys():
            init_params_dict[key] = local_arguments[key]
        return init_params_dict

    def get_config(self) -> dict:
        """
            Method used to get model configuration when saving a model
            :return: Dictionary with arguments required by the class initializer
        """
        return self._init_params
