#coding:utf-8
"""

大多数情况下，您将能够使用高级功能，但有时您可能想要在较低的级别工作。例如，如果您想要实现一个新特性—一些新的内容，那么TensorFlow还没有包括它的高级实现，
比如LSTM中的批处理规范化——那么您可能需要知道一些事情。

这个版本的网络的几乎所有函数都使用tf.nn包进行编写，并且使用tf.nn.batch_normalization函数进行标准化操作

'fully_connected'函数的实现比使用tf.layers包进行编写的要复杂得多。然而，如果你浏览了Batch_Normalization_Lesson笔记本，事情看起来应该很熟悉。
为了增加批量标准化，我们做了如下工作:
Added the is_training parameter to the function signature so we can pass that information to the batch normalization layer.
1.在函数声明中添加'is_training'参数，以确保可以向Batch Normalization层中传递信息
2.去除函数中bias偏置属性和激活函数
3.添加gamma, beta, pop_mean, and pop_variance等变量
4.使用tf.cond函数来解决训练和预测时的使用方法的差异
5.训练时，我们使用tf.nn.moments函数来计算批数据的均值和方差，然后在迭代过程中更新均值和方差的分布，并且使用tf.nn.batch_normalization做标准化
  注意：一定要使用with tf.control_dependencies...语句结构块来强迫Tensorflow先更新均值和方差的分布，再使用执行批标准化操作
6.在前向传播推导时(特指只进行预测，而不对训练参数进行更新时)，我们使用tf.nn.batch_normalization批标准化时其中的均值和方差分布来自于训练时我们
  使用滑动平均算法估计的值。
7.将标准化后的值通过RelU激活函数求得输出
8.不懂请参见https://github.com/udacity/deep-learning/blob/master/batch-norm/Batch_Normalization_Lesson.ipynb
  中关于使用tf.nn.batch_normalization实现'fully_connected'函数的操作
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)


def fully_connected(prev_layer, num_units, is_training):
    """
    num_units参数传递该层神经元的数量，根据prev_layer参数传入值作为该层输入创建全连接神经网络。

   :param prev_layer: Tensor
        该层神经元输入
    :param num_units: int
        该层神经元结点个数
    :param is_training: bool or Tensor
        表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
    :returns Tensor
        一个新的全连接神经网络层
    """

    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)

    gamma = tf.Variable(tf.ones([num_units]))
    beta = tf.Variable(tf.zeros([num_units]))

    pop_mean = tf.Variable(tf.zeros([num_units]), trainable=False)
    pop_variance = tf.Variable(tf.ones([num_units]), trainable=False)

    epsilon = 1e-3

    def batch_norm_training():
        batch_mean, batch_variance = tf.nn.moments(layer, [0])

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)


"""
我们对conv_layer卷积层的改变和我们对fully_connected全连接层的改变几乎差不多。
然而也有很大的区别，卷积层有多个特征图并且每个特征图在输入图层上共享权重
所以我们需要确保应该针对每个特征图而不是卷积层上的每个节点进行Batch Normalization操作

为了实现这一点，我们做了与fully_connected相同的事情，有两个例外:

1.将gamma、beta、pop_mean和pop_方差的大小设置为feature map(输出通道)的数量，而不是输出节点的数量。
2.我们改变传递给tf.nn的参数。时刻确保它计算正确维度的均值和方差。
"""


def conv_layer(prev_layer, layer_depth, is_training):
    """
       使用给定的参数作为输入创建卷积层
        :param prev_layer: Tensor
            传入该层神经元作为输入
        :param layer_depth: int
            我们将根据网络中图层的深度设置特征图的步长和数量。
            这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
        :param is_training: bool or Tensor
            表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
        :returns Tensor
            一个新的卷积层
        """
    strides = 2 if layer_depth%3 == 0 else 1

    in_channels = prev_layer.get_shape().as_list()[3]
    out_channels = layer_depth*4

    weights = tf.Variable(
        tf.truncated_normal([3, 3, in_channels, out_channels], stddev=0.05))

    layer = tf.nn.conv2d(prev_layer, weights, strides=[1, strides, strides, 1], padding='SAME')

    gamma = tf.Variable(tf.ones([out_channels]))
    beta = tf.Variable(tf.zeros([out_channels]))

    pop_mean = tf.Variable(tf.zeros([out_channels]), trainable=False)
    pop_variance = tf.Variable(tf.ones([out_channels]), trainable=False)

    epsilon = 1e-3

    def batch_norm_training():
        # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
        batch_mean, batch_variance = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)


"""
为了修改训练函数，我们需要做以下工作:
1.Added is_training, a placeholder to store a boolean value indicating whether or not the network is training.
添加is_training，一个用于存储布尔值的占位符，该值指示网络是否正在训练
2.Each time we call run on the session, we added to feed_dict the appropriate value for is_training.
每次调用sess.run函数时，我们都添加到feed_dict中is_training的适当值用以表示当前是正在训练还是预测
3.We did not need to add the with tf.control_dependencies... statement that we added in the network that used tf.layers.batch_normalization
because we handled updating the population statistics ourselves in conv_layer and fully_connected.
我们不需要将train_opt训练函数放进with tf.control_dependencies... 的函数结构体中,这是只有在使用tf.layers.batch_normalization才做的更新均值和方差的操作

"""


def train(num_batches, batch_size, learning_rate):
    # Build placeholders for the input samples and labels
    # 创建输入样本和标签的占位符
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    # Add placeholder to indicate whether or not we're training the model
    # 创建占位符表明当前是否正在训练模型
    is_training = tf.placeholder(tf.bool)

    # Feed the inputs into a series of 20 convolutional layers
    # 把输入数据填充到一系列20个卷积层的神经网络中
    layer = inputs
    for layer_i in range(1, 20):
        layer = conv_layer(layer, layer_i, is_training)

    # Flatten the output from the convolutional layers
    # 将卷积层输出扁平化处理
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1]*orig_shape[2]*orig_shape[3]])

    # Add one fully connected layer
    # 添加一个具有100个神经元的全连接层
    layer = fully_connected(layer, 100, is_training)

    # Create the output layer with 1 node for each
    # 为每一个类别添加一个输出节点
    logits = tf.layers.dense(layer, 10)

    # Define loss and training operations
    # 定义loss 函数和训练操作
    model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    # Create operations to test accuracy
    # 创建计算准确度的操作
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train and test the network
    # 训练并测试网络模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_i in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # train this batch
            # 训练样本批次
            sess.run(train_opt, {inputs: batch_xs, labels: batch_ys, is_training: True})

            # Periodically check the validation or training loss and accuracy
            # 定期检查训练或验证集上的loss和精确度
            if batch_i%100 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: mnist.validation.images,
                                                              labels: mnist.validation.labels,
                                                              is_training: False})
                print(
                    'Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            elif batch_i%25 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys, is_training: False})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

        # At the end, score the final accuracy for both the validation and test sets
        # 最后在验证集和测试集上对模型准确率进行评分
        acc = sess.run(accuracy, {inputs: mnist.validation.images,
                                  labels: mnist.validation.labels,
                                  is_training: False})
        print('Final validation accuracy: {:>3.5f}'.format(acc))
        acc = sess.run(accuracy, {inputs: mnist.test.images,
                                  labels: mnist.test.labels,
                                  is_training: False})
        print('Final test accuracy: {:>3.5f}'.format(acc))

        # Score the first 100 test images individually, just to make sure batch normalization really worked
        # 对100个独立的测试图片进行评分,对比验证Batch Normalization的效果
        correct = 0
        for i in range(100):
            correct += sess.run(accuracy, feed_dict={inputs: [mnist.test.images[i]],
                                                     labels: [mnist.test.labels[i]],
                                                     is_training: False})

        print("Accuracy on 100 samples:", correct/100)


num_batches = 800  # 迭代次数
batch_size = 64  # 批处理数量
learning_rate = 0.002  # 学习率

tf.reset_default_graph()
with tf.Graph().as_default():
    train(num_batches, batch_size, learning_rate)


"""
再一次，批量标准化的模型很快达到了很高的精度。
但是在我们的运行中，注意到它似乎并没有学习到前250个批次的任何东西，然后精度开始上升。
这只是显示——即使是批处理标准化，给您的网络一些时间来学习是很重要的。

PS:再100个单个数据的预测上达到了较高的精度，而这才是BN算法真正关注的！！
"""
# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# 2018-03-18 19:35:28.568404: I D:\Build\tensorflow\tensorflow-r1.4\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX
# Batch:  0: Validation loss: 0.69113, Validation accuracy: 0.10020
# Batch: 25: Training loss: 0.57341, Training accuracy: 0.07812
# Batch: 50: Training loss: 0.45526, Training accuracy: 0.04688
# Batch: 75: Training loss: 0.37936, Training accuracy: 0.12500
# Batch: 100: Validation loss: 0.34601, Validation accuracy: 0.10700
# Batch: 125: Training loss: 0.34113, Training accuracy: 0.12500
# Batch: 150: Training loss: 0.33075, Training accuracy: 0.12500
# Batch: 175: Training loss: 0.34333, Training accuracy: 0.15625
# Batch: 200: Validation loss: 0.37085, Validation accuracy: 0.09860
# Batch: 225: Training loss: 0.40175, Training accuracy: 0.09375
# Batch: 250: Training loss: 0.48562, Training accuracy: 0.06250
# Batch: 275: Training loss: 0.67897, Training accuracy: 0.09375
# Batch: 300: Validation loss: 0.48383, Validation accuracy: 0.09880
# Batch: 325: Training loss: 0.43822, Training accuracy: 0.14062
# Batch: 350: Training loss: 0.43227, Training accuracy: 0.18750
# Batch: 375: Training loss: 0.39464, Training accuracy: 0.37500
# Batch: 400: Validation loss: 0.50557, Validation accuracy: 0.25940
# Batch: 425: Training loss: 0.32337, Training accuracy: 0.59375
# Batch: 450: Training loss: 0.14016, Training accuracy: 0.75000
# Batch: 475: Training loss: 0.11652, Training accuracy: 0.78125
# Batch: 500: Validation loss: 0.06241, Validation accuracy: 0.91280
# Batch: 525: Training loss: 0.01880, Training accuracy: 0.96875
# Batch: 550: Training loss: 0.03640, Training accuracy: 0.93750
# Batch: 575: Training loss: 0.07202, Training accuracy: 0.90625
# Batch: 600: Validation loss: 0.03984, Validation accuracy: 0.93960
# Batch: 625: Training loss: 0.00692, Training accuracy: 0.98438
# Batch: 650: Training loss: 0.01251, Training accuracy: 0.96875
# Batch: 675: Training loss: 0.01823, Training accuracy: 0.96875
# Batch: 700: Validation loss: 0.03951, Validation accuracy: 0.94080
# Batch: 725: Training loss: 0.02886, Training accuracy: 0.95312
# Batch: 750: Training loss: 0.06396, Training accuracy: 0.87500
# Batch: 775: Training loss: 0.02013, Training accuracy: 0.98438
# Final validation accuracy: 0.95820
# Final test accuracy: 0.95780
# Accuracy on 100 samples: 0.98