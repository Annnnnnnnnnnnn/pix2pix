"""
工具类
"""
import tensorflow as tf
import numpy as np
import random


def namespace(default_name: str):
    """
    variable space 装饰器
    :param default_name: 待装饰函数
    :return:
    """
    def deco(fn):
        def wrapper(*args, **kwargs):
            if "name" in kwargs.keys():
                name = kwargs["name"]
                kwargs.pop("name")

            else:
                name = default_name

            with tf.variable_scope(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco


class ImagePool(object):
    """
    数据池
    用以装载固定量的数据，并提供获取全部及随机获取一个的途径
    """
    def __init__(self, pool_size: int=50):
        self.pool_size = pool_size
        self._images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        elif len(self._images) < self.pool_size:
            self._images.append(image)
            return image

        else:
            p = random.random()
            if p > 0.5:
                # 使用历史图片
                random_id = random.randrange(0, self.pool_size, step=1)
                temp = self._images[random_id].copy()
                self._images[random_id] = image
                return temp

            else:
                return image


def visual_grid(X: np.array, shape: tuple((int, int))):
    """
    将X中的图片平铺放入新的numpy.array中，用于可视化
    :param X: 图片集合(numpy.array)
    :param shape: 表格形状（行，列）图片数
    :return: 合成后图片array
    """
    nh, nw = shape
    h, w = X.shape[1:3]
    img = np.zeros(shape=(h * nh, w * nw, 3))

    for n, x in enumerate(X):
        i = n % nh
        j = n // nh

        if n >= nh * nw:
            break

        img[i*h:i*h+h, j*w:j*w+w, :] = x

    return img

