"""
可能用到的层或ops
"""
import tensorflow as tf
from GANs.pix2pix.utils import *


@namespace("batch_norm")
def batch_norm(x: tf.Tensor):
    """
    批数据正则
    此处不进行trainning控制，改而使用变量搜集后，在反向传播时限制训练变量进行控制
    :param x:
    :return:
    """
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        scale=True,
                                        is_training=True,
                                        updates_collections=tf.GraphKeys.UPDATE_OPS)


@namespace("conv2d")
def conv2d(x: tf.Tensor, filters: int, kernels: int=4, strides: int=2, stddev: float=0.02, use_bias: bool=False, padding: str="SAME"):
    """
    卷积层
    :param x: Tensor
    :param filters: filter数量
    :param kernels: kernel大小（正方形kernel）
    :param strides: 步长
    :param stddev: 初始化标准差
    :param use_bias: 是否使用偏置
    :param padding: 补零方式（VALID、SAME）
    :return: Tensor
    """
    w = tf.get_variable("w", shape=[kernels, kernels, x.get_shape()[-1], filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)

    if use_bias:
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(conv, b)

    return conv


@namespace("conv2dTranspose")
def deconv2d(x: tf.Tensor, output_shape: tuple, kernels: int=4, strides: int=2, stddev=0.02, use_bias: bool=False, padding: str="SAME"):
    """
    转置卷积
    :param x: Tensor
    :param output_shape: filter数量
    :param kernels: kernel大小（正方形kernel）
    :param strides: 步长
    :param stddev: 初始化标准差
    :param use_bias: 是否使用偏置
    :param padding: 补零方式（VALID、SAME）
    :return: Tensor
    """
    w = tf.get_variable("w", shape=[kernels, kernels, output_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, strides, strides, 1], padding=padding)

    if use_bias:
        b = tf.get_variable("b", shape=[output_shape[-1]], initializer=tf.constant_initializer(0.))
        deconv = tf.nn.bias_add(deconv, b)

    return deconv


@namespace("leaky_relu")
def lrelu(x: tf.Tensor, leak: float=0.2):
    """
    Leaky ReLU激活函数
    :param x: Tensor
    :param leak: 负轴的泄露系数
    :return: Tensor
    """
    return tf.maximum(x, tf.multiply(x, leak))


@namespace("constant_pad")
def constant_pad(x: tf.Tensor, size: int=1):
    """
    以边框临近位置的内容进行补边, 填充0
    :param x: Tensor
    :param size: 边框大小
    :return: Tensor
    """
    return tf.pad(x, paddings=[[0, 0], [size, size], [size, size], [0, 0]], mode="CONSTANT")


@namespace("abs_criterion")
def abs_criterion(in_: tf.Tensor, target: tf.Tensor):
    return tf.reduce_mean(tf.abs(in_ - target))


@namespace("mae_criterion")
def mae_criterion(in_, target):
    return tf.reduce_mean(tf.square(in_ - target))