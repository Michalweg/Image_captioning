from __future__ import annotations

import inspect
import tensorflow as tf


class ImageEncoder(tf.keras.Model):
    """
    Encoder part of the model that uses Inception architecture as its backbone.
    """

    def __init__(self, embedding_dim):
        super(ImageEncoder, self).__init__()
        inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet'
        )
        # Inception architecture without final linear layer.
        self.backbone = tf.keras.Model(inception.input,
                                       inception.layers[-1].output)
        self.reshape_layer = tf.keras.layers.Reshape((-1, self.backbone.output_shape[3]))
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self._init_params = self.get_init_arguments(locals())

    def call(self, inputs):
        """
        Defines a forward pass of an encoder through the model
        :param inputs: Data to be forwarded through the model
        :return: The result of passing data through the model
        """
        inputs = self.backbone(inputs)
        inputs = self.reshape_layer(inputs)
        inputs = self.fc(inputs)
        return inputs

    def get_init_arguments(self, local_arguments: dict) -> dict:
        """
        Extracts arguments that are passed to the class initializer.
        :param local_arguments: Dictionary with all local arguments from the class initializer
        :return: Dictionary with all required arguments from the class initializer
        """
        init_params_dict = dict(inspect.signature(self.__init__).parameters)
        for key in init_params_dict.keys():
            init_params_dict[key] = local_arguments[key]
        return init_params_dict

    def get_config(self):
        """
        Method used to get model configuration when saving a model
        :return: Dictionary with arguments required by the class initializer
        """
        return self._init_params

    @classmethod
    def from_config(cls, config) -> ImageEncoder:
        """
        Method used to extract models parameters when loading a model
        :param config: Dictionary with configuration information
        :return: An object of the class created from the passed config
        """
        return cls(**config)
