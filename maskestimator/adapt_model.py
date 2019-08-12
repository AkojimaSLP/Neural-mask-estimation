# -*- coding: utf-8 -*-
'''
NMBF adaptation
reference:    
The Hitachi/JHU CHiME-5 system:
Advances in speech recognition for everyday home
environments using multiple microphone arrays [Kanda, 2018]    
'''

import soundfile as sf
import numpy as np
import glob
import random
import os

from . import model, feature, shaper

MAX_SEQUENCE = 5000

class adapt_model:
    def __init__(self,
                 model_path,
                 truncate_grad,
                 number_of_stack,
                 lr,
                 spec_dim,
                 sampling_frequency,
                 fftl,
                 shift,
                 left_context,
                 right_contect,
                 number_of_skip_frame,
                 adapt_data_location):
        self.sampling_frequency = sampling_frequency
        self.fftl = fftl
        self.shift = shift
        self.left_context = left_context
        self.right_context = right_contect
        self.number_of_skip_frame = number_of_skip_frame
        self.adapt_data_location = adapt_data_location
        self.lr = lr
        self.model_path = model_path
        self.truncate_grad = truncate_grad
        self.number_of_stack = number_of_stack
        self.spec_dim = spec_dim        

    def get_data_list(self):        
        data_list = []
        file_list = glob.glob(self.adapt_data_location + '/**')
        for ii in range(0, len(file_list)):
            if 'sp_mask_' in file_list[ii]:
                data_list.append(file_list[ii])
        return data_list
        
        
    def create_data_for_adaptation(self,
                                   is_target,
                                   speaker_uttearnce_list) :
        ''' data shape is 
            training data for adaptation : (B, F)
            input data for adaptation: (B, Truncate, T)'''    
            
        mask_estimator_generator = model.NeuralMaskEstimation(self.truncate_grad, self.number_of_stack, self.lr, self.spec_dim)
        mask_estimator = mask_estimator_generator.get_model(is_stateful=True, is_show_detail=False, is_adapt=False)
        mask_estimator = mask_estimator_generator.load_weight_param(mask_estimator, self.model_path)
        
        
        f = open(speaker_uttearnce_list, 'r', encoding='utf-8')
        wav_path = f.readlines()
        f.close()
        
        feature_extractor = feature.Feature(self.sampling_frequency, self.fftl, self.shift)
        data_shaper = shaper.Shape_data(self.left_context,
                          self.right_context,
                          self.truncate_grad,
                          self.number_of_skip_frame)      
        print('creating data for adaptation')
        for wav in wav_path:
            data = sf.read(wav.replace('\n', ''), dtype='float32')[0]
            if len(np.shape(data)) >= 2:
                data = data[:, 0]
            noisy_spectrogram = feature_extractor.get_feature(data)
            noisy_spectrogram = (np.flipud(noisy_spectrogram))    
            noisy_spectrogram = feature_extractor.apply_cmvn(noisy_spectrogram)                
            features = data_shaper.convert_for_predict(noisy_spectrogram)
            print(np.shape(features))
            features_padding, original_batch_size = data_shaper.get_padding_features(features)
            mask_estimator.reset_states()     
            prefix = os.path.splitext(wav)[1]
            print(np.shape(features_padding))
            sp_mask, n_mask = mask_estimator.predict(features_padding, batch_size=MAX_SEQUENCE)
            sp_mask = sp_mask[:original_batch_size, :]
            n_mask = n_mask[:original_batch_size, :]
            save_path_target = self.adapt_data_location + '/' + os.path.basename(wav).replace(prefix, 'sp_mask_' + str(np.int(is_target)) )            
            save_path_input = self.adapt_data_location + '/' + os.path.basename(wav).replace(prefix, 'amp_spec_' + str(np.int(is_target)) )
            save_path_target = save_path_target.replace('\n', '')
            save_path_input = save_path_input.replace('\n', '')            
            np.save(save_path_input, np.array(features))            
            np.save(save_path_target, np.array(sp_mask) * np.int(is_target))            
        print('done.')

    def save_adapt_model(self, save_name):
        ''' data shape is 
            training data for adaptation : (B, F)
            input data for adaptation: (B, Truncate, T)'''            
        mask_estimator_generator = model.NeuralMaskEstimation(self.truncate_grad, self.number_of_stack, self.lr, self.spec_dim)
        mask_estimator = mask_estimator_generator.get_model(is_stateful=False, is_show_detail=False, is_adapt=True)
        mask_estimator = mask_estimator_generator.load_weight_param(mask_estimator, self.model_path)
        
        # ===========================
        #  get wav list for adaptation
        # ===========================
        training_list = self.get_data_list()

        # ===========================
        #  fature dump
        # ===========================
        target_mask = np.zeros((1, self.spec_dim))
        input_amp = np.zeros((1, self.truncate_grad, self.spec_dim))
        for ii in range(0, len(training_list)):
            target_mask = np.concatenate((target_mask, np.load(training_list[ii])), axis=0)
            input_amp = np.concatenate((input_amp, np.load(training_list[ii].replace('sp_mask_', 'amp_spec_'))), axis=0)
        target_mask = target_mask[1:, :]
        input_amp = input_amp[1:, :, :]
        
        # ===========================
        #  fit
        # =========================== 
        shuffle_index = random.sample(range(0, np.shape(target_mask)[0]), np.shape(target_mask)[0])
        print('adaptation...')
        history = mask_estimator.train_on_batch(x=input_amp[shuffle_index, :, :],
                           y=[target_mask[shuffle_index, :],
                              target_mask[shuffle_index, :]])
        print('Done.', history)
        mask_estimator.save_weights(save_name)
        print('save done.' + str(save_name))
