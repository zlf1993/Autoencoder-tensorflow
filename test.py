import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import struct
import os
from tqdm import tqdm
from tensorflow.python.client import device_lib
def load_mnist_train(path, kind='train'): 
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath: 
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16)) 
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels
    
def load_mnist_test(path, kind='t10k'): 
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind) 
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind) 
    with open(labels_path, 'rb') as lbpath: 
        magic, n = struct.unpack('>II',lbpath.read(8)) 
        labels = np.fromfile(lbpath,dtype=np.uint8) 
    with open(images_path, 'rb') as imgpath: 
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16)) 
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels
path = os.getcwd()
train_images, train_labels = load_mnist_train(path)
test_images, test_labels = load_mnist_test(path)
print(len(train_images))
train_epochs = 35
batch_size = 500
noise_factor = 0.5

Input_height = 28
Input_width = 28

input_x = tf.placeholder(tf.float32, shape=[None, Input_height * Input_width], name='input')
input_matrix = tf.reshape(input_x, shape=[-1, Input_height, Input_width, 1])


#encode_processing
#input 28*28
#output 14*14*32
kernal_1 = tf.Variable(tf.truncated_normal(shape=[3,3,1,32], stddev=0.1, name='kernal_1'))
bias_1 = tf.Variable(tf.constant(0.0, shape=[32], name='bias_1'))
conv1 = tf.nn.conv2d(input=input_matrix, filter=kernal_1, strides=[1,1,1,1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')
activ_1 = tf.nn.relu(conv1, name='activ_1')
pool1 = tf.nn.max_pool(value=activ_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool_1')
#pool1 = tf.layers.batch_normalization(pool1)


#input 14*14*32
#output 7*7*32
kernal_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1, name='kernal_2'))
bias_2 = tf.Variable(tf.constant(0.0, shape=[32], name='bias_2'))
conv2 = tf.nn.conv2d(input=pool1, filter=kernal_2, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')
activ_2 = tf.nn.relu(conv2, name='activ_2')
pool2 = tf.nn.max_pool(value=activ_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')
#pool2 = tf.layers.batch_normalization(pool2)

#input 7*7*32
#output 4*4*16
kernal_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 16], stddev=0.1, name='kernal_3'))
bias_3 = tf.Variable(tf.constant(0.0, shape=[16]))
conv3 = tf.nn.conv2d(input=pool2, filter=kernal_3, strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, bias_3)
activ_3 = tf.nn.relu(conv3, name='activ_3')
pool3 = tf.nn.max_pool(value=activ_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')
#pool3 = tf.layers.batch_normalization(pool3)

#input 4*4*16
#output 1*1*64
kernal_4 = tf.Variable(tf.truncated_normal(shape=[4, 4, 16, 64], stddev=0.1, name='kernal_4'))
bias_4 = tf.Variable(tf.constant(0.0, shape=[64]))
conv4 = tf.nn.conv2d(input=pool3, filter=kernal_4, strides=[1, 1, 1, 1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, bias_4)
activ_4 = tf.nn.relu(conv4, name='activ_4')
pool4 = tf.nn.max_pool(value=activ_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_4')


#decode_processing
deconv_weight_0 = tf.Variable(tf.truncated_normal(shape=[4, 4, 16, 64], stddev=0.1), name='deconv_weight_0')
deconv0 = tf.nn.conv2d_transpose(value=pool4, filter=deconv_weight_0, output_shape=[batch_size, 4, 4, 16], strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')


#input 4*4*16
#output 7*7*32
deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 16], stddev=0.1), name='deconv_weight_1')
deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=[batch_size, 7, 7, 32], strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')

 
## 2 deconv layer
## input 7*7*32
## output 14*14*32
deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3,32, 32], stddev=0.1), name='deconv_weight_2')
deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[batch_size, 14, 14, 32], strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
 

## 3 deconv layer
## input 14*14*32
## output 28*28*32
deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1, name='deconv_weight_3'))
deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[batch_size, 28, 28, 32], strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')


##CONV Layer
##input 28*28*32
##output 28*28*1
weight_final = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 1], stddev=0.1, name = 'weight_final'))
bias_final = tf.Variable(tf.constant(0.0, shape=[1], name='bias_final'))
conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')
conv_final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')

output = tf.reshape(conv_final, shape=[-1,Input_height*Input_width])

loss = tf.reduce_mean(tf.pow(tf.subtract(output, input_x), 2.0))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
    #if os.path.exists('tmp/checkpoint'):
    #    saver.restore(sess, 'tmp/model.ckpt')
    #else:
    print("in1")
    sess.run(tf.global_variables_initializer())
    total_batch  = int(len(train_images)/batch_size)
    min_lose = float('inf')
    #min_lose = float('inf')
    for e in range(train_epochs):
        for b in tqdm(range(total_batch)):
            print("in2")
            batch = train_images[b*batch_size:(b+1)*batch_size]
            imgs = batch.reshape((-1,28,28,1))
            batch_cost, _=sess.run([loss, optimizer],feed_dict={input_x:batch})
            msg = "Global Step={:d}, Local batch={:d}, lose={:.4f}"
            if(batch_cost<min_lose):
                min_lose = batch_cost
                #saver.save(sess, 'tmp/model.ckpt')
            #t.set_description("Global Step=%i, local batch=%i, lose=%f" % e % b % batch_cost)
