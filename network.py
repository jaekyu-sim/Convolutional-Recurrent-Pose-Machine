
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def block_vgg_19(input_image):
    #input => 356x356x3
    #block1
    net = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#356x356x64
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#356x356x64
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2))#114x114x64

    #block2
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#178x178x128
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#178x178x128
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2))#89x89x128

    #block3
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#89x89x256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#89x89x256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#89x89x256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#89x89x256
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2))#44x44x256

    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#44x44x512
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#44x44x512

    return net
def block_3d_vgg_19(input_image):
    #input => num_frame x batch x 356 x 356 x 3
    #block1
    net = tf.layers.conv3d(inputs=input_image, filters=64, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x356x356x64
    net = tf.layers.conv3d(inputs=net, filters=64, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x356x356x64
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2))#114x114x64

    #block2
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x178x178x128
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x178x178x128
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2))#89x89x128

    #block3
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x89x89x256
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x89x89x256
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x89x89x256
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x89x89x256
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2))#44x44x256

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#2x44x44x512
    net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))#filters-512
    net = tf.nn.relu(net)
    #print("before flatten : ", np.shape(net))#2x44x44x512
    #net = tf.reshape(net, [-1, 60, 14*14*32])
    #print("after flatten : ", np.shape(net))

    return net

def block_stage_1_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 34
    #print(np.shape(net))
    return net

def block_stage_2_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 34
    #print(np.shape(net))
    return net
def block_stage_3_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 34
    #print(np.shape(net))
    return net
def block_stage_4_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 34
    #print(np.shape(net))
    return net
def block_stage_5_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 34
    #print(np.shape(net))
    return net
def block_stage_6_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 34
    #print(np.shape(net))
    return net

def block_stage_1_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 17
    #print(np.shape(net))
    return net

def block_stage_2_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 17
    #print(np.shape(net))
    return net
def block_stage_3_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 17
    #print(np.shape(net))
    return net
def block_stage_4_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 17
    #print(np.shape(net))
    return net
def block_stage_5_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 17
    #print(np.shape(net))
    return net
def block_stage_6_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 44, 44, 17
    #print(np.shape(net))
    return net


# In[ ]:


def block_3d_stage_1_branch1(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=34, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 34
    print(np.shape(net))
    return net
def block_3d_stage_2_branch1(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=34, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 34
    print(np.shape(net))
    return net
def block_3d_stage_3_branch1(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=34, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 34
    print(np.shape(net))
    return net
def block_3d_stage_4_branch1(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=34, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 34
    print(np.shape(net))
    return net
def block_3d_stage_5_branch1(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=34, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 34
    print(np.shape(net))
    return net
def block_3d_stage_6_branch1(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=34, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 34
    print(np.shape(net))
    return net
def block_3d_stage_1_branch2(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(3, 3, 3), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=17, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 17
    #print(np.shape(net))
    return net
def block_3d_stage_2_branch2(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=17, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 17
    #print(np.shape(net))
    return net
def block_3d_stage_3_branch2(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=17, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 17
    #print(np.shape(net))
    return net
def block_3d_stage_4_branch2(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=17, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 17
    #print(np.shape(net))
    return net
def block_3d_stage_5_branch2(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=17, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 17
    print(np.shape(net))
    return net
def block_3d_stage_6_branch2(feature):
    net = tf.layers.conv3d(inputs=feature, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(7, 7, 7), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv3d(inputs=net, filters=17, kernel_size=(1, 1, 1), padding="SAME", strides=(1, 1, 1))
    #net -> 2, 44, 44, 17
    #print(np.shape(net))
    return net


# In[ ]:


def mobilenet(input_image):
    #block1
    width_multiplier = 1
    net = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_stride, depth_multiplier=1,                                                      kernel_size=[3, 3])

        bn = slim.batch_norm(depthwise_conv)
        pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1])
        bn = slim.batch_norm(pointwise_conv)
        return bn


    net = _depthwise_separable_conv(net, 64, width_multiplier)
    net = _depthwise_separable_conv(net, 64, width_multiplier, downsample=True)
    net = _depthwise_separable_conv(net, 128, width_multiplier)
    net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True)
    net = _depthwise_separable_conv(net, 256, width_multiplier)
    net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True)
    net = _depthwise_separable_conv(net, 256, width_multiplier)
    net = _depthwise_separable_conv(net, 256, width_multiplier)
    net = _depthwise_separable_conv(net, 512, width_multiplier)
    net = _depthwise_separable_conv(net, 512, width_multiplier)
    
    #44x44x512

    return net

