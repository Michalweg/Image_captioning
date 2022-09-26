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

    def call(self, x):
        x = self.backbone(x)
        x = self.reshape_layer(x)
        x = self.fc(x)
        return x
