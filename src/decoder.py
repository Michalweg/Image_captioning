import inspect

import tensorflow as tf

from src.attention import BahdanauAttention


class CaptionDecoder(tf.keras.Model):
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

    def call(self, x):
        x, features, hidden = x
        # defining attention as a separate model
        context_vector, attention_weights = self.attention((features, hidden))

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def get_config(self):
        return self._init_params

    def get_init_parameters(self, local_parameters):
        init_params_dict = dict(inspect.signature(self.__init__).parameters)
        for key in init_params_dict.keys():
            init_params_dict[key] = local_parameters[key]
        return init_params_dict

    @classmethod
    def reset_state(cls, batch_size, units):
        return tf.zeros((batch_size, units))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
