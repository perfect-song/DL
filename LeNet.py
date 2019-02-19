import tensorflow as tf
import numpy as np
import load_data

# hello = tf.constant('hello tensorflow')
# sess = tf.Session()
# print(sess.run(hello))
#
# train_data = load_data.GetTrainDataByLable(b'data')
# print(train_data.shape)

# test_data = load_data.GetTestData(b'data')
# print(test_data.shape)



##input = [32,32,3]
def inference(input_tensor):
    ##第一层 卷积+池化
    w1 = tf.get_variable("w1", shape=[5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))##输入时3通道的，卷积核大小5*5 输出维度32
    b1 = tf.get_variable("b1", shape=[32], initializer=tf.constant_initializer(0.0))
    con1 = tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(con1, b1))
    print("卷积1", relu1.shape)
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print("池化1", pool1.shape)

    ##第二层
    w2 = tf.get_variable("w2", shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", shape=[64], initializer=tf.constant_initializer(0.0))
    con2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(con2, b2))
    print("卷积2",relu2.shape)
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME")
    print("池化2", pool2.shape)

    ##将数据拉平
    pool_shape = pool2.get_shape().as_list()##[3,8,8,64]
    # pool_shape = pool2.get_shape.as_list()
    print(pool_shape)
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]##8*8*34 = 4096

    reshape = tf.reshape(pool2, [pool_shape[0],nodes])##[batch， 4096]

    ##第三层 全连接层 输入4096 输出1024
    w3 = tf.get_variable("w3", shape=[nodes,1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.get_variable("b3",shape=[1024],initializer=tf.constant_initializer(0.0))
    relu3 = tf.nn.relu(tf.matmul(reshape,w3)+b3)
    print("全连接1", relu3.shape)

    ##第四层 全连接层 输入1024 输出512
    w4 = tf.get_variable("w4", shape=[1024,512], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b4 = tf.get_variable("b4",shape=[512],initializer=tf.constant_initializer(0.0))
    relu4 = tf.nn.relu(tf.matmul(relu3,w4)+b4)
    print("全连接2", relu4.shape)

    ##第五层 全连接层 输入515 输出10
    w5 = tf.get_variable("w5",shape=[512,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
    b5 = tf.get_variable("b5", shape=[10], initializer=tf.constant_initializer(0.0))
    output = tf.matmul(relu4,w5)+b5
    print(output.shape)
    return output

# input = train_data[0:3]
# # print(input.shape)
# # print(type(input[0][0][0][0]))
# input_tensor= tf.convert_to_tensor(input)##将numpy转成tensor
# # print(input_tensor.shape)
# # print(type(input_tensor[0][0][0][0]))
#
# output = inference(input_tensor)


def train(data,label):
    x = tf.placeholder(tf.float32, [10, 32, 32, 3], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-output')

    y = inference(x)
    print(y.shape)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    learn_rate = tf.train.exponential_decay(0.01,3000,250,0.1,staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy_mean)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()##参数初始化
        for i in range(5000):
            xs = data[i*10:10*(i+1)]
            ys = label[i*10:10*(i+1)]
            # print(xs.shape)
            # print(ys.shape)
            _, loss,acc = sess.run([train_step, cross_entropy_mean,accuracy],feed_dict={x: xs, y_: ys})
            if i%10==0:
                print("After %d training step(s), loss on training batch is %f, accuracy on training batch is %f.", (i, loss, acc))


train_data = load_data.GetTrainDataByLable(b'data')
train_label = load_data.GetTrainDataByLable(b'labels')
train(train_data, train_label)










