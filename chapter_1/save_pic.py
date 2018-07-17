#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 22:35
# @Author  : ZHANG Haichao
# @File    : save_pic.py
# @Software: PyCharm

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

# 读取MNIST数据集
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# 将原始图片保存在MNIST_data/raw/文件夹下
save_dir='MNIST_data/raw/'

if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前5张
for i in range(5):
    image_array=mnist.train.images[i,:]
    image_array=image_array.reshape(28,28)
    filename=save_dir+'mnist_train_%d.jpg'%i

    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)