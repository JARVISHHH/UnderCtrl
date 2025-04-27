from keras_cv.src.backend import keras

import tensorflow as tf

def timestep_embedding(timesteps, dim=320, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
    freqs = tf.math.exp(
        -log_max_period * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.reshape(tf.cast(timesteps, tf.float32), [-1, 1])* freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], -1)
    return embedding

class PaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

class ZeroPaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)