import inspect

import tensorflow as tf


class ImageEncoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(ImageEncoder, self).__init__()
        inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet'
        )
        self.reshape_layer = tf.keras.layers.Reshape((-1, 2048))
        self.backbone = tf.keras.Model(inception.input,
                                       inception.layers[-1].output)
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self._init_params = self.get_init_parameters(locals())

    def call(self, x):
        x = self.backbone(x)
        x = self.reshape_layer(x)
        x = self.fc(x)
        return x

    def get_init_parameters(self, local_parameters):
        init_params_dict = dict(inspect.signature(self.__init__).parameters)
        for key in init_params_dict.keys():
            init_params_dict[key] = local_parameters[key]
        return init_params_dict

    def get_config(self):
        return self._init_params

    @classmethod
    def from_config(cls, config):
        return cls(**config)
