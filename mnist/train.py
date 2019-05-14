# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import forward

from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100                # 批大小
LEARING_RATE_BASE = 0.8         # 最初学习率
LEARING_RATE_DECAY = 0.99       # 学习速率衰减率
REGULARAZTION_RATE = 0.0001     # 正则化系数
TRAINING_STEPS = 30000          # 训练步数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率
# 模型保存路径及文件名
MODEL_SAVE_PATH = "/home/wangyinzhi/study/TensorFlow/mnist/model/"

MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入输出占位
    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE], name="y-input")

    # 正则公式
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 前向传播
    y = forward.forward(x, regularizer)
    # 训练次数
    global_step = tf.Variable(0, trainable=False)

    # 创建滑动平均值计算器
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算损失值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    """ 计算学习率 exponential_decay(learning_rate 初始学习率,global_step 当前迭代次数, decay_steps 衰减速度,
        decay_rate 学习率衰减系数通常介于0-1之间)
        learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) 
     """
    learning_rate = tf.train.exponential_decay(LEARING_RATE_BASE, global_step, 
                                               mnist.train.num_examples / BATCH_SIZE,
                                              LEARING_RATE_DECAY)
    # 参数更新
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    """tf.control_dependencies是tensorflow中的一个flow顺序控制机制，作用有二：插入依赖（dependencies）和清空依赖（依赖是op或tensor）。常见的tf.control_dependencies是tf.Graph.control_dependencies的装饰器，它们用法是一样的。
        https://blog.csdn.net/hustqb/article/details/83545310
        这里: 在执行train_op之前会先执行[train_step, variable_averages_op]
    """
    with tf.control_dependencies([train_step, variable_averages_op]) : 
        # 空的执行 什么也不做。仅用作控制边的占位符。
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        tf.global_variables_initializer().run()
        # 训练时不再使用验证数据来测试模型，验证和测试通过eval来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
            if i % 1000 == 0 or i == TRAINING_STEPS-1:
                # 训练情况输出，只输出模型在当前训练batch上的损失函数大小
                print("i:%d After %d training steps, loss on training batch is %g" % (i, step, loss_value))
                print(loss_value)
                # 保存当前模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

    print("train stop")
