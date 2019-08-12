# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:32:49 2019

@author: a-kojima
"""
import numpy as np
import os
import soundfile as sf
import pyroomacoustics as pra

class Generate_random_noise:
    def __init__(self,
                 noise_file_list,
                 sampling_frequency):
        self.noise_file_list = noise_file_list
        self.sampling_frequency = sampling_frequency
        
    def get_noise(self, speech_length):
        selected_index = np.random.randint(0, len(self.noise_file_list), size=1)[0]
        selected_noise_path = self.noise_file_list[selected_index]   
        noise_length = np.int(np.float(os.path.basename(selected_noise_path).split('_')[-1].split('.wav')[0])) * self.sampling_frequency
        if noise_length > speech_length:
            noise_cut_start = np.random.randint(0, noise_length - speech_length, size=1)[0]
            noise_data = sf.read(selected_noise_path, dtype='float32', start=noise_cut_start, stop=noise_cut_start + speech_length)[0]
            if len(np.shape(noise_data)) >= 2:
                noise_data = noise_data[:, 0]     
        else:            
            noise_data = sf.read(selected_noise_path)[0]           
            if len(np.shape(noise_data)) >= 2:
                noise_data = noise_data[:, 0]                                                        
            noise_data = np.repeat(noise_data, 30) # adhock-number
            noise_data = noise_data[0:speech_length]
        return noise_data
                        
class SNR_adjusting:
    def __init__(self, speech_data, noise_data):
        self.speech_data = speech_data
        self.noise_data = noise_data
        
    def adjust_SNR(self, speech_data, speech_rate):
        return speech_data * speech_rate
    
    def add_speech_to_noise(self, target_SNR):        
        speech_data = self.normalize_amplitude(self.speech_data, 0.65)
        noise_data = self.normalize_amplitude(self.noise_data, 0.65)
        speech_power_coeficient = self.get_speech_rate(speech_data, noise_data, target_SNR)        
        return (self.adjust_SNR(speech_power_coeficient, speech_data), noise_data)
    
    def normalize_amplitude(self, speech_data, max_amplitude):
        return speech_data/np.max(np.abs(speech_data)) * max_amplitude      
    
    def get_speech_rate(self, speech, noise, target_SNR):
        return 10 ** (target_SNR / np.float(20)) * (np.sum(noise ** 2) / np.float(np.sum(speech ** 2)))   

    def avoid_clipping(self, speech, noise):
        max_amp = (0.9 - 0.01) * np.random.rand() + 0.01
        if (np.max(np.abs(speech))) >= (np.max(np.abs(noise))):
            rate = max_amp / (np.max(np.abs(speech)))
            speech = speech * rate
            noise = noise * rate
        else:
            rate = (max_amp / (np.max(np.abs(noise))))
            noise = noise * rate
            speech = speech * rate
        return speech, noise
    
class RIR_convolve:    
    ''' generate speech using image-method based room simulator 
    '''
    def __init__(self, sampling_frequency)    :
        self.sampling_frequency = sampling_frequency
    
    def get_reverbant_speech(self, speech):
        meters = np.random.randint(5, 7, 1)[0]
        distance = np.random.randint(3, 5, 1)[0]
        rt = (0.5 - 0.01) * np.random.rand() + 0.01
        room = pra.ShoeBox([meters, meters], fs=self.sampling_frequency, t0=0., absorption=rt, max_order=12)
        R = pra.circular_2D_array(center=[distance, distance], M=1, phi0=0, radius=0.07)
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
        room.add_source([1, 1], signal=speech)
        room.simulate()
        ori_length = len(speech)
        speech = room.mic_array.signals.T
        speech = speech[0:ori_length, 0]                
        return speech        