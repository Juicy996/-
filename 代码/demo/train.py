import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("Kannada_MNIST/",one_hot=True)
"""
#输入图片是28*28
n_inputs = 28#输入一行的长度 取决于图片一行28像素
max_time = 28#一共28行
lstm_size = 100#100个隐层单元
n_classes = 10#10个分类
batch_size = 50#每批次50样本
n_batch = mnist.train.num_examples // batch_size#计算一共多少批次 

#这里的none表示第一个维度可以任意，取决于放多少张图片
x = tf.placeholder(tf.float32,[None,784])
#正确的标签
y = tf.placeholder(tf.float32,[None,10])

#初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
#初始化偏置值
biases = tf.Variable(tf.constant(0.1,shape=[n_classes]))

#定义RNN网络
def RNN(X,Weights,Biases):
    #inputs = [batch_size,max_time,n_inputs]
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    #定义LSTM基本CELL
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    #final_state[0]是cell state
    #final_state[1]是hidden_state
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1],Weights)+Biases)
    return results
#计算RNN的返回结果
prediction = RNN(x,weights,biases)
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#把结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy=" + str(acc))

"""







#利用卷积神经网络实现手写数字识别
#mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
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
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print("Iter: " + str(epoch) + ", acc: " + str(acc))









"""
#手写数字识别分类问题，MNIST数据集分类简单版本

#载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
# 批次的大小
batch_size = 128
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 创建神经网络
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([1, 2000]))
# 激活层
layer1 = tf.nn.relu(tf.matmul(x,W1) + b1)
# drop层
layer1 = tf.nn.dropout(layer1,keep_prob=keep_prob)

# 第二层
W2 = tf.Variable(tf.truncated_normal([2000,500],stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 500]))
layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
layer2 = tf.nn.dropout(layer2,keep_prob=keep_prob)

# 第三层
W3 = tf.Variable(tf.truncated_normal([500,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([1,10]))
prediction = tf.nn.sigmoid(tf.matmul(layer2,W3) + b3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)#得到97的正确率
#train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)


# 初始化变量
init = tf.global_variables_initializer()

prediction_2 = tf.nn.softmax(prediction)
# 得到一个布尔型列表，存放结果是否正确
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction_2,1)) #argmax 返回一维张量中最大值索引

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            # 获得批次数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.8})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0} )
        print("Iter " + str(epoch) + " Testing Accuracy: " + str(acc))
"""



"""        
#载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)


#每个批次的大小
batch_size=128
#计算一共有多少个批次
n_batch=mnist.train.num_examples // batch_size
#定义两个placeholder
#定义命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob-input')

#创建一个简单的神经网络
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,w)+b)

#二次代价函数
#loss=tf.reduce_mean(tf.square(y-prediction)) #reduce_mean求平均值

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法进行训练的优化器
optimizer =tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train_step=optimizer.minimize(loss)
#结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) #argmax返回一维张量中最大值所在的位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast转化数据类型

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(31):
        for n_batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter" +str(epoch)+"  Testing Accuracy  " +str(test_acc)+" Training acc "+str(train_acc))


"""



"""
#非线性回归
#使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.1)
#最小化代价函数
train_step=optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    #获取预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
"""



"""
#tensorflow简单实例，训练线性模型
#使用numpy生成100个随机的点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

#构造一个线性模型
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

#二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train=optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))

"""


"""
#fetch
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
add=tf.add(input2,input3)
mul=tf.multiply(input1,add)
#feed
#创建占位符
input4=tf.placeholder(tf.float32)
input5=tf.placeholder(tf.float32)
output=tf.multiply(input4,input5)
with tf.Session() as sess:
    #在一个session中运行多个op
    result=sess.run([mul,add])
    print(result)
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input4:7.,input5:2.}))

"""



"""
#创建变量
w1=tf.Variable(0,name='con')
w2=tf.add(w1,1)
#赋值操作
updata=tf.assign(w1,w2)
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    for _ in range(5):
        sess.run(updata)
        print(sess.run(w1))
    
 """   


"""
a='ffffugggggooo44ppko'
a=[np.random.normal(0, 0.1, 50).tolist(), np.random.normal(0, 0.1, 50).tolist()]
print(a)
a.append(np.random.normal([1,3]).tolist())
print(a)
"""

"""
b=collections.Counter(a)
d=b.most_common(4)
e=dict(b.most_common(4))
c=len(b)
print(b)
print(c)
print(d)
print(e)
print(e.values())
print(e.keys())
print(len(e))
"""