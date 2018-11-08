
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import ast
import cv2
import json
import network_rnn as net
import math
import collections
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread


# In[2]:


img_width = 112#112
network_output_width = 14#14
vgg_3d_channel = 25
def video_anno_data_using_make_file_batch(video_path, anno_path, step_size = 60):
    batch_size = len(video_path)
    width = img_width
    height = img_width
    channel = 3
    
    video = []
    anno = []
    
    for i in range(batch_size):
        #print("hello")
        anno_tmp, video_tmp = load_data(anno_path[i], video_path[i])
        #rint("hello")
        anno.append(anno_tmp)
        video.append(video_tmp)
        
    return video, anno
        
def make_file_batch(batch_size = 1):
    video_path = './video_with_abstacle/TrainingSet_2/3/'
    anno_path = './video_with_abstacle/TrainingSet_2/4/'
    
    video_list = os.listdir(video_path)
    anno_list = os.listdir(anno_path)
    
    video_list.sort()
    anno_list.sort()
    
    num_of_data = len(video_list)
    index = np.arange(0, num_of_data)
    np.random.shuffle(index)
    index = index[:batch_size]
    
    shuffled_video_path = [video_path +'/'+ video_list[i] for i in index]
    shuffled_anno_path = [anno_path +'/'+ anno_list[j] for j in index]
    
    return np.asarray(shuffled_video_path), np.asarray(shuffled_anno_path)
def make_test_batch(batch_size = 1):
    video_path = './video_with_abstacle/TrainingSet_2/3'
    anno_path = './video_with_abstacle/TrainingSet_2/4'
    
    video_list = os.listdir(video_path)
    anno_list = os.listdir(anno_path)
    
    video_list.sort()
    anno_list.sort()
    
    num_of_data = len(video_list)
    index = np.arange(0, num_of_data)
    np.random.shuffle(index)
    index = index[:batch_size]
    
    shuffled_video_path = [video_path +'/'+ video_list[i] for i in index]
    shuffled_anno_path = [anno_path +'/'+ anno_list[j] for j in index]
    print("test video : ",shuffled_video_path)
    return np.asarray(shuffled_video_path), np.asarray(shuffled_anno_path)

def _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
    start = stride / 2.0 - 0.5

    center_x, center_y = joint

    for g_y in range(height):
        for g_x in range(width):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x-center_x) * (x-center_x) + (y-center_y) * (y-center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue

            heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
            if heatmap[g_y, g_x, plane_idx] > 1.0:
                heatmap[g_y, g_x, plane_idx] = 1.0
                
                
def _put_paf_on_plane(vectormap, countmap, plane_idx, center_from, center_to, threshold, height, width, stride):
    center_from = (center_from[0] // stride, center_from[1] // stride)
    center_to = (center_to[0] // stride, center_to[1] // stride)

    vec_x = center_to[0] - center_from[0]
    vec_y = center_to[1] - center_from[1]

    min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
    min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

    max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
    max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 1e-8: #1e-8 이하는 0으로 인식되서 0으로 나눌수 없다는 에러 발생. 따라서 return처리 해줌
        return

    vec_x /= norm
    vec_y /= norm
    
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            bec_x = x - center_from[0]
            bec_y = y - center_from[1]
            dist = abs(bec_x * vec_y - bec_y * vec_x)

            if dist > threshold:
                continue

            countmap[x][y][plane_idx] = countmap[x][y][plane_idx] + 1

            vectormap[x][y][plane_idx*2+0] = vec_x
            vectormap[x][y][plane_idx*2+1] = vec_y
            

def bubble_sort(L):
    for i in range(len(L)-1):
        for j in range(len(L)-1):
            if L[j] > L[j+1]:
                temp = L[j+1]
                L[j+1] = L[j]
                L[j] = temp

def load_data(anno_path, video_path):#아직은 annotation data만 load
    #Annotation Data Load
    f = open(anno_path)
    
    s = f.readlines()

    file_name = []
    parts = []
    _joint_data = []

    
    file_path = []
    joint_data = []
    for index, data in enumerate(s):
        #print(i)
        raw_data = ast.literal_eval(data)
        #print(raw_data)
        raw_data = json.loads(raw_data)
        joint_data.append(raw_data)
    #print(joint_data[0])
    
    heatmap_height = network_output_width
    heatmap_width = network_output_width
    index = 0
    for j in joint_data:
        #print(j)
        height = img_width
        width = img_width
        if(j == {}):
            pass
        else:
            array = j.keys()
            #print(array)
            #print(j)
            for k in array:
                joint_data[index][k][0] = int(j[k][0] / height / 3.17 * heatmap_height)
                joint_data[index][k][1] = int(j[k][1] / width / 3.17 * heatmap_width)#3.17 -> 356/112
                #print(j[k][0], j[k][1])
        index = index + 1
    
    #Video Data Load
    cap = cv2.VideoCapture(video_path)
    video_data = []
    i = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(i != length):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (img_width, img_width))
        video_data.append(frame)
        i = i + 1
    return joint_data, video_data


# In[3]:


class openpose():
    def __init__(self, batch_size, sess):
        self.sess = sess
        num_frame = 60
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, num_frame, img_width, img_width, 3])
        self.confidence_map_label = tf.placeholder(dtype=tf.float32, shape=[None, num_frame, network_output_width, network_output_width, 17])
        self.vector_map_label = tf.placeholder(dtype=tf.float32, shape=[None, num_frame, network_output_width, network_output_width, 34])
        self.batch_size = batch_size
        print("test data load start")
        #self.test_image_path = self.load_test_data()
        print("test data load finish")

        self.model()
        self.optimizer()
        print("open pose init complete")
        
    def load_test_data(self):
        img_path = "./MPII_Dataset/resized_test_image/"
        file_path = []
        file_list = os.listdir(img_path)
        for i in (file_list):
            file_path.append(img_path + i)
        return file_path
        
    
    def make_heatmap(self, anno_data, width = network_output_width, height = network_output_width, num_of_maps = 17):
        batch_size = len(anno_data)
        width = network_output_width
        height = network_output_width
        num_of_maps = 17

        output = np.zeros((batch_size, width, height, num_of_maps))
        for index, joints in enumerate(anno_data):
            #print(len(anno_data))
            #print(index)
            #joint_data = anno_data
            heatmap = np.zeros((width, height, num_of_maps), np.int16)#batch 일단 뺌
            #print(joints)

            buffer = list(joints.items())
            key_buffer = joints.keys()

            for i in range(len(buffer)):
                buffer[i] = list(buffer[i])
                buffer[i][0] = int(buffer[i][0])
            bubble_sort(buffer)

            idx = 0
            for j in range(17):
                if('%d' %j in key_buffer):
                    center_x = buffer[idx][1][0]
                    center_y = buffer[idx][1][1]
                    #joint = buffer[idx][1]
                    joint = [center_y, center_x]
                    idx = idx + 1
                    _put_heatmap_on_plane(heatmap, plane_idx = j, joint = joint, sigma = 3, height = height,                                           width = width, stride = 1)
                else:
                    pass
            idx = 0
            heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)
            output[index] = heatmap
            #print("make_heatmap's output shape : ", np.shape(output))
            #output -> frame, width, height, channel
            output2 = np.transpose(output, [0, 3, 1, 2])
        return output2
        
    
    def make_paf_field(self, batch_joint_data, width = network_output_width, height = network_output_width, num_of_maps = 17):
        batch_size = len(batch_joint_data)
        output1 = np.zeros((batch_size, width, height, num_of_maps*2))
        output2 = np.zeros((batch_size, width, height, num_of_maps))
        for index, joints in enumerate(batch_joint_data):
            joint_pairs = list(zip(
                [9, 8, 8, 8,13,14,12,11,7,6,6,3,4,2,1],
                [8,13,12, 7,14,15,11,10,6,3,2,4,5,1,0]))
            #make vector map
            width = network_output_width
            height = network_output_width
            num_of_maps = 17
            vectormap = np.zeros((width, height, num_of_maps*2), dtype=np.float32)#batch 일단 뺌
            countmap = np.zeros((width, height, num_of_maps), np.int16)#batch 일단 뺌

            key = (joints.keys())
            for plane_idx, (j_idx1, j_idx2) in enumerate(joint_pairs):
                if(('%d' %j_idx1 in key) and ('%d' %j_idx2 in key)):

                    center_from = joints['%d'%j_idx1]
                    center_to = joints['%d'%j_idx2]

                    if not center_from or not center_to:
                        continue
                    _put_paf_on_plane(vectormap=vectormap, countmap=countmap, plane_idx=plane_idx, center_from=center_from, center_to=center_to,                                       threshold=1, height=network_output_width, width=network_output_width, stride = 1)

            nonzeros = np.nonzero(countmap)


            for x, y, p in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
                if countmap[x][y][p] <= 0:
                    continue
                vectormap[x][y][p*2+0] /= countmap[x][y][p]
                vectormap[x][y][p*2+1] /= countmap[x][y][p]

            output1[index] = vectormap.astype(np.float32)
            output2[index] = countmap
            #print("make_paf's output shape : ", np.shape(output1))
        return output1, output2 #output1 -> vectormap, output2 -> countmap


        
    
    def model(self):

        #(self.X)
        stage0_data = net.block_3d_vgg_19(self.X)#stage0_data - None, frame, 14, 14, 32
        #stage0_data 's shape -> None, 60, 14*14*16
        self.vgg_output = stage0_data
        hidden_size = network_output_width*network_output_width*vgg_3d_channel

        cell = tf.contrib.rnn.GRUCell(hidden_size)
        initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        
        reshaped_stage0_data = tf.reshape(stage0_data, [-1, 60, network_output_width*network_output_width*vgg_3d_channel])
        output, state = tf.nn.dynamic_rnn(cell = cell, inputs = reshaped_stage0_data,                                           dtype = tf.float32, initial_state=initial_state)
        
        print("RNN output's shape : ", np.shape(output))
        #tf.contrib.
        stage0_data = tf.reshape(output, [-1, 60, network_output_width, network_output_width, vgg_3d_channel])
        print("RNN reshaped output's shape : ", np.shape(stage0_data))

        self.stage1_branch1 = net.block_3d_stage_1_branch1(stage0_data)#stage1_branch1 - None, frame, 44, 44, 34        
        self.stage1_branch2 = net.block_3d_stage_1_branch2(stage0_data)#stage1_branch2 - None, frame, 44, 44, 17
        self.stage1_data = tf.concat([self.stage1_branch1, self.stage1_branch2, stage0_data], 4)
        ##########################################################################################################
        self.stage2_branch1 = net.block_3d_stage_2_branch1(self.stage1_data)#stage2_branch1 - None, frame, 44, 44, 34
        self.stage2_branch2 = net.block_3d_stage_2_branch2(self.stage1_data)#stage2_branch2 - None, frame, 44, 44, 17
        self.stage2_data = tf.concat([self.stage2_branch1, self.stage2_branch2, stage0_data], 4)
        ##########################################################################################################
        self.stage3_branch1 = net.block_3d_stage_3_branch1(self.stage2_data)#stage2_branch1 - None, frame, 44, 44, 34
        self.stage3_branch2 = net.block_3d_stage_3_branch2(self.stage2_data)#stage2_branch2 - None, frame, 44, 44, 17
        self.stage3_data = tf.concat([self.stage3_branch1, self.stage3_branch2, stage0_data], 4)
        ##########################################################################################################
        self.stage4_branch1 = net.block_3d_stage_4_branch1(self.stage3_data)#stage2_branch1 - None, frame, 44, 44, 34
        self.stage4_branch2 = net.block_3d_stage_4_branch2(self.stage3_data)#stage2_branch2 - None, frame, 44, 44, 17
        self.stage4_data = tf.concat([self.stage4_branch1, self.stage4_branch2, stage0_data], 4)
        ##########################################################################################################
        self.stage5_branch1 = net.block_3d_stage_5_branch1(self.stage4_data)#stage2_branch1 - None, frame, 44, 44, 34
        self.stage5_branch2 = net.block_3d_stage_5_branch2(self.stage4_data)#stage2_branch2 - None, frame, 44, 44, 17
        self.stage5_data = tf.concat([self.stage5_branch1, self.stage5_branch2, stage0_data], 4)
        ##########################################################################################################
        self.stage6_branch1 = net.block_3d_stage_6_branch1(self.stage5_data)#stage2_branch1 - None, frame, 44, 44, 34
        self.stage6_branch2 = net.block_3d_stage_6_branch2(self.stage5_data)#stage2_branch2 - None, frame, 44, 44, 17
        self.stage6_data = tf.concat([self.stage6_branch1, self.stage6_branch2, stage0_data], 4)
        ##########################################################################################################
        #self.stage7_branch1 = net.block_3d_stage_7_branch1(self.stage6_data)#stage2_branch1 - None, frame, 44, 44, 34
        #self.stage7_branch2 = net.block_3d_stage_7_branch2(self.stage6_data)#stage2_branch2 - None, frame, 44, 44, 17
        #self.stage7_data = tf.concat([self.stage7_branch1, self.stage7_branch2, stage0_data], 4)
        ##########################################################################################################
        #self.stage8_branch1 = net.block_3d_stage_8_branch1(self.stage7_data)#stage2_branch1 - None, frame, 44, 44, 34
        #self.stage8_branch2 = net.block_3d_stage_8_branch2(self.stage7_data)#stage2_branch2 - None, frame, 44, 44, 17
        #self.stage8_data = tf.concat([self.stage8_branch1, self.stage8_branch2, stage0_data], 4)
        ##########################################################################################################
        #self.stage9_branch1 = net.block_3d_stage_9_branch1(self.stage8_data)#stage2_branch1 - None, frame, 44, 44, 34
        #self.stage9_branch2 = net.block_3d_stage_9_branch2(self.stage8_data)#stage2_branch2 - None, frame, 44, 44, 17
        #self.stage9_data = tf.concat([self.stage9_branch1, self.stage9_branch2, stage0_data], 4)
        ##########################################################################################################
    def optimizer(self):
        
        W_p = 1
        #loss는 euclid loss 함수 사용
        #affinity field loss - branch1
        self.loss_stage1_branch1 = tf.nn.l2_loss(W_p*(self.stage1_branch1 - self.vector_map_label)) / self.batch_size
        self.loss_stage2_branch1 = tf.nn.l2_loss(W_p*(self.stage2_branch1 - self.vector_map_label)) / self.batch_size
        self.loss_stage3_branch1 = tf.nn.l2_loss(W_p*(self.stage3_branch1 - self.vector_map_label)) / self.batch_size
        self.loss_stage4_branch1 = tf.nn.l2_loss(W_p*(self.stage4_branch1 - self.vector_map_label)) / self.batch_size
        self.loss_stage5_branch1 = tf.nn.l2_loss(W_p*(self.stage5_branch1 - self.vector_map_label)) / self.batch_size
        self.loss_stage6_branch1 = tf.nn.l2_loss(W_p*(self.stage6_branch1 - self.vector_map_label)) / self.batch_size
        #self.loss_stage7_branch1 = tf.nn.l2_loss(W_p*(self.stage7_branch1 - self.vector_map_label)) / self.batch_size
        #self.loss_stage8_branch1 = tf.nn.l2_loss(W_p*(self.stage8_branch1 - self.vector_map_label)) / self.batch_size
        #self.loss_stage9_branch1 = tf.nn.l2_loss(W_p*(self.stage9_branch1 - self.vector_map_label)) / self.batch_size
        
        #confidence map loss - branch2
        self.loss_stage1_branch2 = tf.nn.l2_loss(W_p*(self.stage1_branch2 - self.confidence_map_label)) / self.batch_size
        self.loss_stage2_branch2 = tf.nn.l2_loss(W_p*(self.stage2_branch2 - self.confidence_map_label)) / self.batch_size
        self.loss_stage3_branch2 = tf.nn.l2_loss(W_p*(self.stage3_branch2 - self.confidence_map_label)) / self.batch_size
        self.loss_stage4_branch2 = tf.nn.l2_loss(W_p*(self.stage4_branch2 - self.confidence_map_label)) / self.batch_size
        self.loss_stage5_branch2 = tf.nn.l2_loss(W_p*(self.stage5_branch2 - self.confidence_map_label)) / self.batch_size
        self.loss_stage6_branch2 = tf.nn.l2_loss(W_p*(self.stage6_branch2 - self.confidence_map_label)) / self.batch_size
        #self.loss_stage7_branch2 = tf.nn.l2_loss(W_p*(self.stage7_branch2 - self.confidence_map_label)) / self.batch_size
        #self.loss_stage8_branch2 = tf.nn.l2_loss(W_p*(self.stage8_branch2 - self.confidence_map_label)) / self.batch_size
        #self.loss_stage9_branch2 = tf.nn.l2_loss(W_p*(self.stage9_branch2 - self.confidence_map_label)) / self.batch_size
                
        #rnn network loss
        
        self.loss1 = tf.reduce_mean([self.loss_stage1_branch1, self.loss_stage1_branch2])
        self.loss2 = tf.reduce_mean([self.loss_stage2_branch1, self.loss_stage2_branch2])
        self.loss3 = tf.reduce_mean([self.loss_stage3_branch1, self.loss_stage3_branch2])
        self.loss4 = tf.reduce_mean([self.loss_stage4_branch1, self.loss_stage4_branch2])
        self.loss5 = tf.reduce_mean([self.loss_stage5_branch1, self.loss_stage5_branch2])
        self.loss6 = tf.reduce_mean([self.loss_stage6_branch1, self.loss_stage6_branch2])
        #self.loss7 = tf.reduce_mean([self.loss_stage7_branch1, self.loss_stage7_branch2])
        #self.loss8 = tf.reduce_mean([self.loss_stage8_branch1, self.loss_stage8_branch2])
        #self.loss9 = tf.reduce_mean([self.loss_stage9_branch1, self.loss_stage9_branch2])
        
        self.total_loss = (self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.loss5 + self.loss6) / self.batch_size
        #self.total_loss = (self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.loss5 + self.loss6 + self.loss7\
        #                  + self.loss8 + self.loss9) / self.batch_size
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 4e-5#4e-5
        lr = tf.train.exponential_decay(starter_learning_rate, global_step,4000, 0.333, staircase=True)#100000->5000
        self.optimizer_total_loss = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.total_loss, global_step=global_step)
        

    def train(self):
        data_size = 1562#41749
        num_video_frame = 60
        batch_size = self.batch_size
        total_batch = int(data_size / batch_size)

        h_loss_data = []
        v_loss_data = []
        graph_loss_iter1 = []
        graph_loss_iter2 = []
        
        SAVE_PATH = "C:/Users/JAEKYU/Documents/Jupyter Notebook/Open_Pose/Weight_rnn/Weight.ckpt"
        print("graph init start")
        self.sess.run(tf.global_variables_initializer())
        print("graph init finish")
        saver = tf.train.Saver()

        try:
            saver.restore(self.sess, SAVE_PATH)
            print("----------------Weight load----------------")
        except:
            print("---------------No Weight Exist---------------")
            print("----------------Training Start---------------")
        
        iteration = 0
        for epoch in range(10000):#15
            print("epoch",epoch+1, "start")
            loss_iter1 = 0
            loss_iter2 = 0
            
            for i in range(10):#total_batch
                #여기 앞에까진 1.json 에 대한 annotation data 생성 완료. batch 를 할 필요
                #data load, batch 생성
                
                _batch_video_path, _batch_anno_path = make_file_batch(batch_size)
                _batch_video_data, _batch_anno_data =                 video_anno_data_using_make_file_batch(_batch_video_path, _batch_anno_path, num_video_frame)
                #_batch_video_data, _batch_anno_data가 training에 사용됨
                #_batch_video_data -> [batch, frame_size, width, height, channel]
                #_batch_anno_data가 -> [batch, ]
                
                #[batch, frame_size, width, height, channel]을 [frame_size, batch, width, height, channel]로 변경
                
                _transposed_batch_video_data = np.transpose(_batch_video_data, [1,0,2,3,4])
                _transposed_batch_anno_data = np.transpose(_batch_anno_data, [1,0])
                
                data = np.zeros((batch_size, num_video_frame, img_width, img_width, 3), dtype=np.float32)
                for batch in range(batch_size):
                    for frame in range(num_video_frame):
                        data[batch][frame] = (_batch_video_data[batch][frame].astype(np.float32) / 256.0)
                        
                _heatmap = []
                _vectormap = []
                _countmap = []
                #data_squeeze = np.squeeze(data)
                for j in range(num_video_frame):
                    heatmap = self.make_heatmap(_transposed_batch_anno_data[j], width=14, height=14, num_of_maps=17)
                    vectormap, countmap = self.make_paf_field(_transposed_batch_anno_data[j], width = 14, height = 14, num_of_maps = 17)

                    _heatmap.append(heatmap)
                    _vectormap.append(vectormap)
                    _countmap.append(countmap)
                #print("_heatmap's shape : ", np.shape(_heatmap))
                transpose_heatmap = np.transpose(_heatmap, [1,0,4,3,2])#3, 4, 2
                #print("transpose_heatmap's shape : ", np.shape(transpose_heatmap))
                transpose_vectormap = np.transpose(_vectormap, [1,0,3,2,4])
                transpose_countmap = np.transpose(_countmap, [1,0,3,2,4])
                
                #vectormap_squeeze = np.squeeze(transpose_vectormap)
                #heatmap_squeeze = np.squeeze(transpose_heatmap)
                #data, transpose_heatmap, transpose_vectormap
                #print("data_squeeze : ", np.shape(data_squeeze))
                #print("vectormap, countmap : ", np.shape(transpose_vectormap), np.shape(transpose_heatmap))
                total_loss_opt, Heat_loss, Vector_loss =                 self.sess.run([self.optimizer_total_loss, self.loss_stage6_branch1, self.loss_stage6_branch2],
                              #_batch_video_data
                         feed_dict = {self.X : data, \
                                      self.confidence_map_label : transpose_heatmap, \
                                      self.vector_map_label : transpose_vectormap})

                h_loss_data.append(Heat_loss)
                v_loss_data.append(Vector_loss)
                loss_iter1 = loss_iter1 + Heat_loss
                loss_iter2 = loss_iter2 + Vector_loss
                iteration = iteration + 1
                
            loss_iter1 = loss_iter1 / 10#total_batch
            loss_iter2 = loss_iter2 / 10#total_batch
            
            graph_loss_iter1.append(loss_iter1)
            graph_loss_iter2.append(loss_iter2)
            
            print("heatmap cost")
            print("Heat_loss : ", Heat_loss)
            print("vectormap cost")
            print("Vector_loss : ", Vector_loss)
            print('\n')
            print("avg'Heat_loss : ", loss_iter1)
            print("avg'Vector_loss : ", loss_iter2)
            

            
            np.savez("./Loss/heat_loss.npz", iteration, graph_loss_iter1)
            np.savez("./Loss/vector_loss.npz", iteration, graph_loss_iter2)
            
            plt.plot(graph_loss_iter1)
            plt.show()
            plt.plot(graph_loss_iter2)
            plt.show()
            plt.plot(h_loss_data)
            plt.show()
            plt.plot(v_loss_data)
            plt.show()
            saver.save(self.sess, SAVE_PATH)
        #return batch_img, batch_img_path, batch_annotation, heatmap, vectormap, countmap
        #return self.image_path, self.annotation_data

        
    def test(self):
        batch_size = self.batch_size
        num_video_frame = 60
        _batch_video_path, _batch_anno_path = make_test_batch(batch_size)
        _batch_video_data, _batch_anno_data =         video_anno_data_using_make_file_batch(_batch_video_path, _batch_anno_path, num_video_frame)
        
        data = np.zeros((batch_size, num_video_frame, img_width, img_width, 3), dtype=np.float32)
        for batch in range(batch_size):
            for frame in range(num_video_frame):
                data[batch][frame] = (_batch_video_data[batch][frame].astype(np.float32) / 256.0)
        
        
        #data_squeeze = np.squeeze(data)
        #vectormap_squeeze = np.squeeze(transpose_vectormap)
        #heatmap_squeeze = np.squeeze(transpose_heatmap)
        
        
        
        SAVE_PATH = "C:/Users/JAEKYU/Documents/Jupyter Notebook/Open_Pose/Weight_rnn/Weight.ckpt"
        print("graph init start")
        self.sess.run(tf.global_variables_initializer())
        print("graph init finish")
        saver = tf.train.Saver()

        saver.restore(self.sess, SAVE_PATH)
        print("----------------Weight load----------------")
        
        paf, heatmap, vgg_output = self.sess.run([self.stage6_branch1, self.stage6_branch2, self.vgg_output], feed_dict = {self.X : data})
        return paf, heatmap, vgg_output, _batch_video_data

    def test_data_set(self):
        num_video_frame = 60
        batch_size = self.batch_size
        
        for epoch in range(1):#15
            print("epoch",epoch+1, "start")
            loss_iter1 = 0
            loss_iter2 = 0
            
            for i in range(1):#total_batch
                #여기 앞에까진 1.json 에 대한 annotation data 생성 완료. batch 를 할 필요
                #data load, batch 생성
                
                _batch_video_path, _batch_anno_path = make_file_batch(batch_size)
                _batch_video_data, _batch_anno_data =                 video_anno_data_using_make_file_batch(_batch_video_path, _batch_anno_path, num_video_frame)
                #_batch_video_data, _batch_anno_data가 training에 사용됨
                #_batch_video_data -> [batch, frame_size, width, height, channel]
                #_batch_anno_data가 -> [batch, ]
                
                #[batch, frame_size, width, height, channel]을 [frame_size, batch, width, height, channel]로 변경
                
                _transposed_batch_video_data = np.transpose(_batch_video_data, [1,0,2,3,4])
                _transposed_batch_anno_data = np.transpose(_batch_anno_data, [1,0])
                
                data = np.zeros((batch_size, num_video_frame, img_width, img_width, 3), dtype=np.float32)
                for batch in range(batch_size):
                    for frame in range(num_video_frame):
                        data[batch][frame] = (_batch_video_data[batch][frame].astype(np.float32) / 256.0)
                        
                _heatmap = []
                _vectormap = []
                _countmap = []
                #data_squeeze = np.squeeze(data)
                for j in range(num_video_frame):
                    heatmap = self.make_heatmap(_transposed_batch_anno_data[j], width=14, height=14, num_of_maps=17)
                    vectormap, countmap = self.make_paf_field(_transposed_batch_anno_data[j], width = 14, height = 14, num_of_maps = 17)

                    _heatmap.append(heatmap)
                    _vectormap.append(vectormap)
                    _countmap.append(countmap)
                #print("_heatmap's shape : ", np.shape(_heatmap))
                transpose_heatmap = np.transpose(_heatmap, [1,0,4,3,2])#3, 4, 2
                #print("_vectormap's shape : ", np.shape(_vectormap))
                transpose_vectormap = np.transpose(_vectormap, [1,0,3,2,4])
                transpose_countmap = np.transpose(_countmap, [1,0,3,2,4])
                
                #vectormap_squeeze = np.squeeze(transpose_vectormap)
                #heatmap_squeeze = np.squeeze(transpose_heatmap)
                
                #print("data_squeeze : ", np.shape(data_squeeze))
                #print("vectormap, countmap : ", np.shape(transpose_vectormap), np.shape(transpose_heatmap))
                
                

        return data, transpose_heatmap, transpose_vectormap, transpose_countmap
        #return self.image_path, self.annotation_data

        


    
print("cell end")


# Training Session

# In[4]:

