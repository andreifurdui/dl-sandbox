import tensorflow.keras.layers as layers
import tensorflow_addons as tfa


class ResNetBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        self.conv_1 = layers.Conv2D(filters=output_res, kernel_size=kernel_size, strides=strides, groups=groups,
                               padding="same", **kwargs)
        self.conv_2 = layers.Conv2D(filters=output_res, kernel_size=kernel_size, strides=1, groups=groups,
                               padding="same", **kwargs)
        self.conv_skip = None
        if strides > 1:
            self.conv_skip = layers.Conv2D(filters=output_res, kernel_size=1, strides=strides)

        self.norm_1 = tfa.layers.InstanceNormalization()
        self.norm_2 = tfa.layers.InstanceNormalization()

        self.act_1 = layers.ELU()
        self.act_2 = layers.ELU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_1(inputs)
        x = self.act_1(x)
        x = self.norm_1(x)

        x = self.conv_2(x)

        skip = self.conv_skip(inputs)
        x += skip

        x = self.act_2(x)
        x = self.norm_2(x)

        return x


class ResNetBlockUp(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        


