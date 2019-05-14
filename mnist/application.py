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
def application(img_path, my_flg):
    with tf.Graph().as_default() as g:
        # 定义输入输出占位
        x = tf.placeholder(tf.float32, [1, forward.INPUT_NODE], name="x-input")

        # 前向传播，因测试时无需关注正则化损失值
        y = forward.forward(x, None)
        value = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        files = os.listdir(img_path)
        img_rs=[]
        img_gs=[]
        img_bs=[]
        lables=[]

        for file in files:
            the_file = os.path.join(img_path, file)
            lables.append(file)
            img_cv = cv2.imread(the_file)
            img_cv = img_cv.astype('float')
            print("img_cv shape:", img_cv.shape)
            cv2.normalize(img_cv, img_cv,1.0, 0.0, cv2.NORM_MINMAX)
            print("img_cv shape:", img_cv.shape)
            img_shrink = cv2.resize(img_cv, (28,28), interpolation=cv2.INTER_LINEAR)
            if "1" == my_flg:
                img_shrink = 1 - img_shrink
            print("img_shrink shape:", img_shrink.shape)
            img_r, img_g, img_b = cv2.split(img_shrink)
            print("img_r shape:", img_r.shape)
            print("img_g shape:", img_g.shape)
            print("img_b shape:", img_b.shape)
            #scipy.misc.toimage(img_r, cmin=0.0,cmax=1.0).save("E:\Study\TensorFlow\data\mnist\\0_shrink.png")

            img_r_reshape = img_r.reshape(1,784)
            img_g_reshape = img_g.reshape(1,784)
            img_b_reshape = img_b.reshape(1,784)
            img_rs.append(img_r_reshape)
            img_gs.append(img_g_reshape)
            img_bs.append(img_b_reshape)
            print("img_r_reshape shape:", img_r_reshape.shape)
            print("img_g_reshape shape:", img_g_reshape.shape)
            print("img_b_reshape shape:", img_b_reshape.shape)

        print(lables)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            # 加载模型
            saver.restore(sess, "E:\Study\TensorFlow\FC\mnist-base-py\model\model.ckpt-30000")
            #saver.restore(sess, "/home/wangyinzhi/study/TensorFlow/mnist/model/model.ckpt-30000")

            size = len(img_bs)

            for i in range(size):
                value_r,y_r = sess.run([value, y], feed_dict={x:img_rs[i]})
                value_g,y_g = sess.run([value, y], feed_dict={x:img_gs[i]})
                value_b,y_b = sess.run([value, y], feed_dict={x:img_bs[i]})

                #print("y_r:", y_r)
                #print("y_g:", y_g)
                #print("y_b:", y_b)

                print("lables:%s  r:%d  g:%d  b:%d" % (lables[i], value_r, value_g, value_b))

def main(argv=None):
    print("argv:", sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    application(sys.argv[1], sys.argv[2])
    print("asdf")

if __name__ == '__main__':
    tf.app.run()
