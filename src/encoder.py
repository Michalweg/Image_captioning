import tensorflow as tf


class ImageEncoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(ImageEncoder, self).__init__()
        inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet'
        )
        self.model = tf.keras.Model(inception.input,
                                    inception.layers[-1].output)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.model(x)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
