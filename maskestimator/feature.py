# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:25:12 2019

@author: a-kojima
"""

import numpy as np
import librosa

class Feature:

    def __init__(self,
                 sampling_frequency,
                 fftl,
                 shift):
        self.sampling_frequency = sampling_frequency
        self.fftl = fftl
        self.shift = shift
        
    def add_white_noise(self, data, min_amp=0.00001):
        return data + np.random.normal(loc=0, scale=min_amp, size=len(data))
    
    def get_feature(self, speech):        
        spectrogram = librosa.core.stft(speech,
                                        n_fft=self.fftl,
                                        hop_length=self.shift,
                                        win_length=self.fftl)        
        return np.abs(spectrogram)
        
    def get_ideal_binary_mask_herman(self,
                                     speech_spectrogram,
                                     noise_spectrogram,
                               threshold_bin=100,
                               theta_sp_low=0,
                               theta_sp_high=0,
                               theta_n_low=0,
                               theta_n_high=0):
        speech_mask = np.sqrt(np.abs(speech_spectrogram) ** 2) / np.sqrt(np.abs(noise_spectrogram) ** 2)
        noise_mask = np.sqrt(np.abs(speech_spectrogram) ** 2) / np.sqrt(np.abs(noise_spectrogram) ** 2)
       
        speech_mask_low = speech_mask[0:threshold_bin, :]
        speech_mask_high = speech_mask[threshold_bin:, :]
        noise_mask_low = noise_mask[0:threshold_bin, :]
        noise_mask_high = noise_mask[threshold_bin:, :]
        speech_mask_low[speech_mask_low > theta_sp_low] = 1
        speech_mask_low[speech_mask_low <= theta_sp_low] = 0        

        speech_mask_high[speech_mask_high > theta_sp_high] = 1
        speech_mask_high[speech_mask_high <= theta_sp_high] = 0        

        noise_mask_low[noise_mask_low > theta_n_low] = 1
        noise_mask_low[noise_mask_low <= theta_n_low] = 0        

        noise_mask_high[noise_mask_high > theta_n_high] = 1
        noise_mask_high[noise_mask_high <= theta_n_high] = 0       
        
        speech_mask[0:threshold_bin, :] = speech_mask_low
        speech_mask[threshold_bin:, :] = speech_mask_high
                
        noise_mask[0:threshold_bin, :] = noise_mask_low
        noise_mask[threshold_bin:, :] = noise_mask_high

        noise_mask_tmp = self.apply_cmvn(speech_spectrogram ** 2)
        speech_mask_tmp = noise_mask_tmp
        noise_mask_tmp[noise_mask_tmp <= 0.0001] = 0
        noise_mask_tmp[noise_mask_tmp > 0.0001] = 1
        
        speech_mask_tmp[speech_mask_tmp <= 0.01] = 0
        speech_mask_tmp[speech_mask_tmp > 0.01] = 1
                
        noise_mask_tmp = 1 - noise_mask_tmp
        noise_mask = np.logical_and(noise_mask_tmp, noise_mask)                        
        speech_mask = np.logical_and(speech_mask_tmp, speech_mask)   
        
        speech_mask, noise_mask = self.apply_filter_spech_component(speech_mask, noise_mask)

        return (speech_mask.astype(np.int), noise_mask.astype(np.int))    

    def apply_cmvn(self, specs):
        mean = np.mean(specs, axis=1)      
        std_var = np.std(specs, axis=1)
        return ((specs.T - mean) / std_var).T    
    
    def apply_range_norm(self, specs):
        specs = ((specs - np.min(specs)) / (np.max(specs) - np.min(specs))) * (1 - 0) + 0
        return specs
    
    def apply_filter_spech_component(self, speech_mask, noise_mask):
        freq_grid = np.linspace(0, self.sampling_frequency, self.fftl)[0:np.int(self.fftl / 2) + 1]
        hz_90_index = np.argmin(np.abs(freq_grid - 50))
        speech_mask[ 0: hz_90_index, :] = 0.0
        noise_mask[ 0: hz_90_index, :] = 1.0
        hz_7800_index = np.argmin(np.abs(freq_grid - 7900))
        speech_mask[  hz_7800_index:, :] = 0.0
        noise_mask[  hz_7800_index:, :] = 1.0
        return speech_mask, noise_mask    
