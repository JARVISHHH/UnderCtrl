from keras_cv.src.backend import keras

class ZeroPaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)