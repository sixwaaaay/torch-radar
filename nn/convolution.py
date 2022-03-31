# convolution.py
# 说明： 卷积相关实现
# Author: [sixwaaaay](https://github.com/sixwaaaay)
from typing import Tuple

import numpy as np


# 将数据转换为矩阵
# 参考： UCBerkeley, https://github.com/BVLC/caffe
def img2col(X, filter_size: Tuple[int, int], stride: int, padding: int):
    """
    :param X: 特征，形状 (batch, channel, height, width)
    :param filter_size: 卷积核大小 (height, width)
    :param stride: 步长
    :param padding: 填充
    :return: (batch, channel, height, width)
    """
    batch, channel, height, width, = X.shape
    filter_height, filter_width = filter_size
    out_height = (height + 2 * padding - filter_height) // stride + 1
    out_width = (width + 2 * padding - filter_width) // stride + 1
    img_col = np.zeros((batch, channel, filter_height, filter_width, out_height, out_width))
    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            img_col[:, :, y, x, :, :] = X[:, :, y:y_max:stride, x:x_max:stride]
    return img_col.transpose([0, 4, 5, 1, 2, 3]).reshape(batch * out_height * out_width, -1)


# 将矩阵转换为数据
# 参考：UCBerkeley, https://github.com/BVLC/caffe
def col2img(col: np.ndarray, features_shape, filter_size: Tuple[int, int], stride: int, padding: int):
    """
    :param col:
    :param features_shape: 形状元组 (batch, channel, height, width)
    :param filter_size: 卷积核大小 (height, width)
    :param stride: 步长
    :param padding: 填充
    :return: (batch, channel, height, width)
    """
    batch, channel, height, width = features_shape
    filter_height, filter_width = filter_size
    out_height = (height + 2 * padding - filter_height) // stride + 1
    out_width = (width + 2 * padding - filter_width) // stride + 1
    col = col.reshape((batch, out_height, out_width, channel, filter_height, filter_width)).transpose(
        [0, 3, 4, 5, 1, 2])
    img = np.zeros((batch, channel, height + 2 * padding, width + 2 * padding))
    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, padding:height + padding, padding:width + padding]


# 步长为1，填充为0的卷积操作
def naive_cov_2d(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
