import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#利用卷积神经网络实现手写数字识别
#mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
mnist=input_data.read_data_sets("Kannada_MNIST",one_hot=True)
batch_size = 1
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1) #生成一个随机的正态分布
    return tf.Variable(initial)
    #return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

#初始化偏置
def bias_vairable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    #return tf.Variable(tf.constant(0.1, shape=shape))

#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义两个人placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#改变x的格式转为4D的向量
x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一个卷积层的权重和偏置
W_conv1 = weight_variable([5,5,1,32]) # 5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = bias_vairable([32]) #每个卷积核一个偏置值

# 28*28*1 的图片卷积之后变为28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化之后变为 14*14*32
h_pool1 = max_pool_2x2(h_conv1)

# 第二次卷积之后变为 14*14*64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_vairable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
# 第二次池化之后变为 7*7*64
h_pool2 = max_pool_2x2(h_conv2)


# 第一个全连接层
W_fc1 = weight_variable([7*7*64,1024]) #上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_vairable([1024]) #1024个节点

# 把池化层第二层的输出扁平化为一维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#h_fc1_drop = tf.nn.dropout(h_fc1)

# 第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_vairable([10])
logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
prediction = tf.nn.sigmoid(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

prediction_2 = tf.nn.softmax(prediction)
#结果存放在一个布尔列表中
correct_prediction = (tf.equal(tf.argmax(prediction_2,1), tf.argmax(y,1)))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:0.7})
        print("Iter: " + str(epoch) + ", acc: " + str(acc))
