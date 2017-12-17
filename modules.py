"""
模块化的生成器与判别器
"""

from GANs.pix2pix.utils import *
from GANs.pix2pix.layers import *


@namespace("Generator")
def generator(x: tf.Tensor, reuse: bool=False, channels: int=64):
    """
    生成器
    :param x: Tensor
    :param reuse: 是否重用变量
    :param channels:
    :return: Tensor
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    layers = []
    with tf.variable_scope("encoder"):
        # encoder_0: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        y = conv2d(x, filters=channels, kernels=4, strides=2, name="conv2d_0")
        layers.append(y)

        layer_specs = [
            channels * 2,  # encoder_1: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            channels * 4,  # encoder_2: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            channels * 8,  # encoder_3: [batch, 32, 32, ngf * 2] => [batch, 16, 16, ngf * 8]
            channels * 8,  # encoder_4: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            channels * 8,  # encoder_5: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            channels * 8,  # encoder_6: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            channels * 8,  # encoder_7: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for i, out_channel in enumerate(layer_specs):
            y = lrelu(layers[-1])
            y = conv2d(y, out_channel, kernels=4, strides=2, name="conv2d_%d" % (i + 1))
            y = batch_norm(y, name="bn_%d" % (i + 1))
            layers.append(y)

    with tf.variable_scope("decoder"):
        layer_specs = [
            (channels * 8, 0.5),  # decoder_7: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
            (channels * 8, 0.5),  # decoder_6: [batch, 2, 2, ngf * 8] => [batch, 4, 4, ngf * 8]
            (channels * 8, 0.5),  # decoder_5: [batch, 4, 4, ngf * 8] => [batch, 8, 8, ngf * 8]
            (channels * 8, 0.),   # decoder_4: [batch, 8, 8, ngf * 8] => [batch, 16, 16, ngf * 8]
            (channels * 4, 0.),   # decoder_3: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 2]
            (channels * 2, 0.),   # decoder_2: [batch, 32, 32, ngf * 2] => [batch, 64, 64, ngf * 2]
            (channels, 0.)        # decoder_1: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        ]

        num_encoder_layers = len(layers)

        for i, (out_channel, dropout) in enumerate(layer_specs):

            skip_layer = num_encoder_layers - i - 1
            if i == 0:
                y = layers[-1]
                output_shape = layers[skip_layer - 1].get_shape().as_list()

            else:
                y = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                output_shape = layers[skip_layer - 1].get_shape().as_list()

            y = tf.nn.relu(y)

            # [batch, in_height, in_width, in_channels] => [batch, in_height * 2, in_width * 2, out_channels]
            y = deconv2d(y, output_shape=output_shape, kernels=4, strides=2, name="conv2dT_%d" % skip_layer)
            y = batch_norm(y, name="bn_%d" % skip_layer)

            if dropout > 0.:
                y = tf.nn.dropout(y, keep_prob=1 - dropout)

            layers.append(y)

        # decoder_0: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        y = tf.concat([layers[-1], layers[0]], axis=3)
        y = tf.nn.relu(y)
        y = deconv2d(y, output_shape=x.get_shape().as_list(), kernels=4, strides=2, name="conv2dT_0")
        y = tf.tanh(y)
        layers.append(y)

        # for i in layers:
        #     print(i.get_shape())

        return layers[-1]


@namespace("Discriminator")
def discriminator(x: tf.Tensor, y: tf.Tensor, reuse: bool=False, channel: int=64):
    """
    判别器
    :param x: Tensor
    :param y: Tensor
    :param reuse: 是否重用变量
    :param channel:
    :return:
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    layers = []
    # 2 x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    z = tf.concat([x, y], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    z = conv2d(z, filters=channel, kernels=4, strides=2, name="conv2d_0")
    z = lrelu(z)
    layers.append(z)
    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(3):
        filters = channel * min(2 ** (i + 1), 8)
        strides = 1 if i == 2 else 2
        z = conv2d(z, filters=filters, kernels=4, strides=strides, name="conv2d_%d" % (i + 1))
        z = batch_norm(z, name="bn_%d" % (i + 1))
        z = lrelu(z)
        layers.append(z)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    z = conv2d(z, filters=1, strides=1, name="conv2d_5")
    z = tf.nn.sigmoid(z)
    layers.append(z)

    # for i in layers:
    #     print(i)

    return z


