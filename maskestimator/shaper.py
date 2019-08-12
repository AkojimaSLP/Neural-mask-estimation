# -*- coding: utf-8 -*-
'''
Created on Wed Aug  7 10:25:44 2019

@author: a-kojima

'''
import numpy as np

MAX_SEQUENCE = 5000

class Shape_data:
    def __init__(self, left_stack, right_stack, max_sequence, number_of_skip_frames):
        self.left_stack = left_stack
        self.right_stack = right_stack
        self.max_sequence = max_sequence
        self.number_of_skip_frames =  number_of_skip_frames
        self.number_of_stack = self.left_stack + self.right_stack + 1
    
    def convert_for_train(self, data, label1, label2):
        '''
        feature: (T, F) -> (B, TRUNCATE, F)
        label: (T, F) -> (B, F)
        '''
        
        stack_data = []
        stack_label_sp = []        
        stack_label_n = []        
        fftl, number_of_frames = np.shape(data)
        number_of_sample = np.int((number_of_frames - (self.left_stack + self.right_stack + 1)) / (self.number_of_skip_frames + 1))  # # number of sample
        number_of_mini_batch = np.int(number_of_sample - self.max_sequence)

        if number_of_mini_batch == 0: # less than 1 block
            return (np.array([]), np.array([]), np.array([]))
        utterance_pointer = self.left_stack     
                       
        for j in range(0, number_of_mini_batch):
            tmp_stack_data = []
            center_position = utterance_pointer
            for i in range(0, self.max_sequence):
                cut_data = data[:, center_position - self.left_stack:center_position + self.right_stack + 1 ]
                if i == np.int(self.max_sequence / 2): 
                    cut_label1 = (label1[:, center_position])
                    cut_label2 = (label2[:, center_position])
                vec_data = np.reshape(cut_data, fftl * self.number_of_stack)
                tmp_stack_data.append(vec_data)
                center_position = center_position + 1 + self.number_of_skip_frames
            stack_data.append(tmp_stack_data)            
            stack_label_sp.append(cut_label1)            
            stack_label_n.append(cut_label2)            
            utterance_pointer = utterance_pointer + 1
        return (stack_data, stack_label_sp, stack_label_n)    

    def convert_for_predict(self, data):
        '''
        feature: (T, F) -> (B, TRUNCATE, F)
        '''        
        stack_data = []
        fftl, number_of_frames = np.shape(data)
        number_of_sample = np.int((number_of_frames - (self.left_stack + self.right_stack + 1)) / (self.number_of_skip_frames + 1))  # # number of sample
        number_of_mini_batch = np.int(number_of_sample - self.max_sequence)

        if number_of_mini_batch == 0: # less than 1 block
            return (np.array([]), np.array([]), np.array([]))

        utterance_pointer = self.left_stack     
                       
        for j in range(0, number_of_mini_batch):
            tmp_stack_data = []
            center_position = utterance_pointer
            for i in range(0, self.max_sequence):
                cut_data = data[:, center_position - self.left_stack:center_position + self.right_stack + 1 ]
                vec_data = np.reshape(cut_data, fftl * self.number_of_stack)
                tmp_stack_data.append(vec_data)
                center_position = center_position + 1 + self.number_of_skip_frames
            stack_data.append(tmp_stack_data)            
            utterance_pointer = utterance_pointer + 1
        return stack_data            
    
    def get_padding_features(self, predict_features):
        batch, sequence, feature_order = np.shape(predict_features)
        padding_feature = np.zeros((MAX_SEQUENCE, sequence, feature_order), dtype=np.float32)
        padding_feature[: batch, :, :] = predict_features
        return padding_feature, batch
