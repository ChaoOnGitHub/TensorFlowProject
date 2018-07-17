#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 16:25
# @Author  : ZHANG Haichao
# @File    : convolutional.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#1. 读入数据
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

##1.1 将输入x转成卷积网络能够识别的形式[-1,28,28,1]，其中-1类似于上面的None，由输入确定
x_image=tf.reshape(x,[-1,28,28,1])

#2. 建立卷积网络
##2.1 构建卷积函数
def weight_Variable(shape):
    # 随机初始化权重
    inital=tf.truncated_normal(shape,mean=0.0,stddev=0.1)#正态分布函数，产生随机值，平均值和标准差
    return tf.Variable(inital)

def bias_Variable(shape):
    # 随机初始化偏置项
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

def conv2d(x,W):
    # 构建卷积函数
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # 构建最大池化函数
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

##2.2 构建卷积网络
###2.2.1 第一层卷积
W_conv1=weight_Variable([5,5,1,32])
b_conv1=bias_Variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

###2.2.2 第二层卷积
W_conv2=weight_Variable([5,5,32,64])
b_conv2=bias_Variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

###2.2.3 全连接层
W_fc1=weight_Variable([7*7*64,1024])
b_fc1=bias_Variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# 使用Dropout，防止过拟合
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

###2.2.4 全连接层
W_fc2=weight_Variable([1024,10])
b_fc2=bias_Variable([10])
y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2


#3. 定义损失函数和优化函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#4. 定义测试集上的准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#5. 训练模型
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch=mnist.train.next_batch(100)

    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})

        print("step :%d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

#6. 测试数据集准确率
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))