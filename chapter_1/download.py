#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 22:20
# @Author  : ZHANG Haichao
# @File    : download.py
# @Software: PyCharm

from tensorflow.examples.tutorials.mnist import input_data

# 从MNIST_data中读取MNIST数据
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# 查看训练集、验证集、测试集的大小
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)

print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# 打印第0幅图片的向量
print(mnist.train.images[0,:])

# 打印第0幅图片的标签
print(mnist.train.labels[0,:])

