# -*- coding:utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    """ 获取变量
        Inputs:
        - shape: 样式
        - regularizer: 惩罚函数
        OutPuts:
        - 参数
        """
    weights = tf.get_variable("weights", shape,initializer = tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights

def forward(input_data, regularizer):
    """ 前向传播
        Inputs:
        - data: 输入数据
        - regularizer: 惩罚函数
        OutPuts:
        - 前向传播结果
    """
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_data, weights) + biases)

    with tf.variable_scope('h1'):
        weights = get_weight_variable([LAYER1_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(h1, weights) + biases

    return layer2
