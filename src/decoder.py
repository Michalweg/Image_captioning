from __future__ import annotations

import inspect
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from src.attention import BahdanauAttention


class CaptionDecoder(tf.keras.Model):
    """
    Decoder part of the model, which generates texts based on the features obtained from the encoder.
    """
    def __init__(self, embedding_dim, units, vocab_size, max_caption_length):
        super(CaptionDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_caption_length)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length
        self._init_params = self.get_init_parameters(locals())

    def call(self, inputs):
        """
        Defines a forward pass of an attention model through the model
        :param inputs: Data to be forwarded through the attention
        :return: The result of passing data through the model
        """
        inputs, features, hidden = inputs
        # defining attention as a separate model
        context_vector, attention_weights = self.attention((features, hidden))

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        inputs = self.embedding(inputs)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        inputs = tf.concat([tf.expand_dims(context_vector, 1), inputs], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(inputs)

        # shape == (batch_size, max_length, hidden_size)
        inputs = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        inputs = tf.reshape(inputs, (-1, inputs.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        inputs = self.fc2(inputs)

        return inputs, state, attention_weights

    def get_config(self) -> dict:
        """
        Method used to get model configuration when saving a model
        :return: Dictionary with arguments required by the class initializer
        """
        return self._init_params

    def get_init_parameters(self, local_arguments) -> dict:
        """
        Extracts arguments that are passed to the class initializer.
        :param local_arguments: Dictionary with all local arguments from the class initializer
        :return: Dictionary with all required arguments from the class initializer
        """
        init_params_dict = dict(inspect.signature(self.__init__).parameters)
        for key in init_params_dict.keys():
            init_params_dict[key] = local_arguments[key]
        return init_params_dict

    @classmethod
    def reset_state(cls, batch_size, units) -> EagerTensor:
        return tf.zeros((batch_size, units))

    @classmethod
    def from_config(cls, config) -> CaptionDecoder:
        """
       Method used to extract models parameters when loading a model
       :param config: Dictionary with configuration information
       :return: An object of the class created from the passed config
       """
        return cls(**config)
