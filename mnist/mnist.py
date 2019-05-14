import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import train
import evaluate
import numpy as np
import threading
import time
import forward
import os
import sys
import cv2
import scipy.misc

# 测试
def test(model_path, mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出占位
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE], name="y-input")
        validate_feed = {x:mnist.test.images, y_:mnist.test.labels}

        # 前向传播，因测试时无需关注正则化损失值
        y = forward.forward(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            if (model_path):
                # 加载模型
                saver.restore(sess, model_path)
                accuracy_score, x1, y1, y_1= sess.run([accuracy, x, y, y_], feed_dict= validate_feed)
                print("x shape:", x1.shape)
                print("y shape:", y1.shape)
                print("y_ shape:", y_1.shape)

                #print("y0:", y1[0])
                #print("y_0:", y_1[0])
                #print("x2:", x1[2])

                #img = x1[0].reshape(28,28)
                #scipy.misc.toimage(img, cmin=0.0,cmax=1.0).save("/home/wangyinzhi/video/mnist_test_0.png")

                #for j in range(50):
                #    img = x1[60+j].reshape(28,28)
                #    file_path = "/home/wangyinzhi/video/mnist_test_"
                #    file_path += str(j)
                #    file_path += ".png"
                #    print(file_path)
                #    scipy.misc.toimage(img, cmin=0.0,cmax=1.0).save(file_path)


                #print("y_0:",  y_[0].eval())
                print("test accuracy_score:%g" % (accuracy_score))
            else:
                print("model file[%s] not find" % (model_path))


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    mnist = input_data.read_data_sets("/home/wangyinzhi/study/data/mnist/train", one_hot=True)
    list = ["/home/wangyinzhi/study/TensorFlow/mnist/model/model.ckpt-30000"]
    print("argv:", sys.argv[1])

    if '1' == sys.argv[1]:
        list[0] = ""
        # 训练
        train_thread = threading.Thread(target=train.train, args=(mnist,))
        train_thread.start()
        ## 验证
        evaluate_thread = threading.Thread(target=evaluate.evaluate, args=(mnist,list,))
        evaluate_thread.start()

        train_thread.join()
        time.sleep(evaluate.EVAL_INTERVAL_SECS)
        evaluate.EXIT_FLAG = 1
        evaluate_thread.join()
    # 测试
    test(list[0], mnist)

if __name__ == '__main__':
    tf.app.run()
