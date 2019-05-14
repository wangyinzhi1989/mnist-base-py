# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载forward和train中的常量和信息
import forward
import train

# 新模型check间隔
EVAL_INTERVAL_SECS = 10
# 最优的模型
FINE_MODEL = ""
# 退出标志
EXIT_FLAG = 0

def evaluate(mnist,list):
    with tf.Graph().as_default() as g:
        # 定义输入输出占位
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE], name="y-input")
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        # 前向传播，因验证时无需关注正则化损失值
        y = forward.forward(x, None)

        """ 使用前向传播结果计算正确率，使用tf.argmax(y,1)对前向结果进行分类预测
            tf.argmax(y, axis)  当axis=0时，以列为单位取该列的最大值索引；
                                当axis=1时，以行为单位取该行最大值索引
            如：[[1,2,3],[2,3,4],[5,4,3],[8,7,2]] axis=0:[3 3 1] axis=1:[2 2 0 0]
        """
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        """ 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用滑动
            平均的函数获取平均值了。这样就可以完全共用前向传播过程了。
            通过使用variables_to_restore函数，可以使在加载模型的时候将影子变量直接
            映射到变量的本身，所以我们在获取变量的滑动平均值的时候只需要获取到变量
            的本身值而不需要去获取影子变量
        """
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 间隔EVAL_INTERVAL_SECS秒加载一次最新的模型，如果是最新的，则进行验证
        old_model = ""
        old_score = 0
        while True:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config = config) as sess:
                # 查找最新文件
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path and ckpt.model_checkpoint_path != old_model:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获取迭代轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict= validate_feed)
                    print("evaluate step[%s] model, validation accuracy ＝ %g" % (global_step, accuracy_score))
                    if (old_score < accuracy_score):
                        list[0] = ckpt.model_checkpoint_path
                        print("fine model[%s]" % (list[0]))
                else:
                    print("not found new model")
                if EXIT_FLAG:
                    print("evaluate stop")
                    return
                # sleep
                time.sleep(EVAL_INTERVAL_SECS)

