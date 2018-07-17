#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 10:20
# @Author  : ZHANG Haichao
# @File    : softmax_regression.py
# @Software: PyCharm

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

## 1. 定义模型
# 创建占位符x，代表待识别的图片
x=tf.placeholder(tf.float32,[None,784])

# W是否softmax模型的参数，将一个784维的输入转换为一个10维的输出
W=tf.Variable(tf.zeros([784,10]))

# b是偏置项
b=tf.Variable(tf.zeros([10]))

# y表示输出
y=tf.nn.softmax(tf.matmul(x,W)+b)

# y_是实际输出
y_=tf.placeholder(tf.float32,[None,10])


## 2. 构造损失函数，这里使用交叉熵损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))


## 3. 构造优化函数，这是使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


## 4. 创建训练会话
sess=tf.InteractiveSession()

# 初始所有变量，分配内存
tf.global_variables_initializer().run()

# 进行梯度下降
for _ in range(1000):
    # 设定minibatch数
    batch_x,batch_y=mnist.train.next_batch(100)

    # 在session里运行train_step
    sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})

## 5. 模型准确率判定
correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_predict,tf.float32))
print("The accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
