# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:40:29 2019

@author: a-kojima
"""
import numpy as np
import copy
import matplotlib.pyplot as pl
from . import util

class complexGMM_mvdr:
    
    def __init__(self,
                 sampling_frequency,
                 fft_length,
                 fft_shift,
                 number_of_EM_iterate,
                 min_segment_dur,
                 condition_number_inv_threshold=10**(-6),
                 scm_inv_threshold=10**(-10),
                 beamformer_inv_threshold=10**(-6)):        
        self.sampling_frequency=sampling_frequency
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        self.number_of_EM_iterate=number_of_EM_iterate
        self.min_segment_dur=min_segment_dur
        self.condition_number_inv_threshold=condition_number_inv_threshold
        self.scm_inv_threshold=scm_inv_threshold
        self.beamformer_inv_threshold=beamformer_inv_threshold
        
    def get_spatial_correlation_matrix(self, speech_data):
        complex_spectrum, _ = util.get_3dim_spectrum_from_data(speech_data, self.fft_length, self.fft_shift, self.fft_length)
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        
        # CGMM parameters
        lambda_noise = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        lambda_noisy = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        phi_noise = np.ones((number_of_frames, number_of_bins), dtype=np.float64)
        phi_noisy = np.ones((number_of_frames, number_of_bins), dtype=np.float64)
        R_noise = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        R_noisy = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)        
        yyh = np.zeros((number_of_channels, number_of_channels, number_of_frames, number_of_bins), dtype=np.complex64)
        
        # init R_noisy and R_noise
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                h = np.multiply.outer(complex_spectrum[:, t, f], np.conj(complex_spectrum[:, t, f]).T)
                yyh[:, :, t, f] = h
                R_noisy[:, :, f] = R_noisy[:, :, f] + h
            R_noisy[:, :, f] = R_noisy[:, :, f] / number_of_frames
            R_noise[:, :, f] = np.eye(number_of_channels, number_of_channels, dtype=np.complex64)        
        R_xn = copy.deepcopy(R_noisy)                    
        p_noise =  np.ones((number_of_frames, number_of_bins), dtype=np.float64)
        p_noisy =  np.ones((number_of_frames, number_of_bins), dtype=np.float64)
            
        # go EMiteration
        for ite in range(0, self.number_of_EM_iterate):
            print('iter', str(ite + 1) + '/' + str(self.number_of_EM_iterate))
            for f in range(0, number_of_bins):                             
                R_noisy_onbin = copy.deepcopy(R_noisy[:, :, f])
                R_noise_onbin = copy.deepcopy(R_noise[:, :, f])                
                if np.linalg.cond(R_noisy_onbin) < self.condition_number_inv_threshold:
                    R_noisy_onbin = R_noisy_onbin + self.condition_number_inv_threshold * np.eye(number_of_channels) * np.max(np.diag(R_noisy_onbin))                    
                if np.linalg.cond(R_noise_onbin) < self.condition_number_inv_threshold:
                    R_noise_onbin = R_noise_onbin + self.condition_number_inv_threshold * np.eye(number_of_channels) * np.max(np.diag(R_noise_onbin))                
            
                R_noisy_inv = np.linalg.pinv(R_noisy_onbin, rcond=self.scm_inv_threshold)
                R_noise_inv = np.linalg.pinv(R_noise_onbin, rcond=self.scm_inv_threshold)                                            
                R_noisy_accu = np.zeros((number_of_channels, number_of_channels), dtype=np.complex64)
                R_noise_accu = np.zeros((number_of_channels, number_of_channels), dtype=np.complex64)
                
                for t in range(0, number_of_frames):
                    corre = yyh[:, :, t, f]                
                    obs = complex_spectrum[:, t, f]    
                    
                    # update phi (real)
                    phi_noise[t, f] = np.real(np.trace(np.matmul(corre, R_noise_inv), dtype=np.float64) / number_of_channels)
                    phi_noisy[t, f] = np.real(np.trace(np.matmul(corre, R_noisy_inv), dtype=np.float64) / number_of_channels)                    
                    if phi_noise[t, f] == 0:
                        phi_noise[t, f] = self.condition_number_inv_threshold
                    if phi_noisy[t, f] == 0:
                        phi_noisy[t, f] = self.condition_number_inv_threshold
                                            
                    # update p (real)
                    k_noise_1 = np.matmul(np.conj(obs).T , R_noise_inv / phi_noise[t, f])            
                    k_noise = np.matmul(k_noise_1, obs)       
                    tmp_p_noise = np.linalg.det((phi_noise[t, f] * R_noise_onbin).astype(np.float64))
                    p_noise[t, f] = np.real(np.exp( - np.real(k_noise).astype(np.float64)) / (np.pi * tmp_p_noise))                    
                    # avoid nan or inf
                    if np.isnan(p_noise[t, f]) == True or np.isinf(p_noise[t, f]) == True:
                        p_noise[t, f] = np.nan_to_num(p_noise[t, f])                    
                    k_noisy_1 = np.matmul(np.conj(obs).T, R_noisy_inv / phi_noisy[t, f])
                    k_noisy = np.real(np.matmul(k_noisy_1, obs))
                    tmp_p_noisy = np.linalg.det((phi_noisy[t, f] * R_noisy_onbin).astype(np.float64))
                    p_noisy[t, f] = np.real(np.exp( - np.real(k_noisy).astype(np.float64)) / (np.pi * tmp_p_noisy))     
                    # avoid nan or inf
                    if np.isnan(p_noisy[t, f]) == True or np.isinf(p_noisy[t, f]) == True:
                        p_noisy[t, f] = np.nan_to_num(p_noisy[t, f])
            
                    # update lambda
                    lambda_noise[t, f] = p_noise[t, f] / (p_noise[t, f] + p_noisy[t, f])
                    lambda_noisy[t, f] = p_noisy[t, f] / (p_noise[t, f] + p_noisy[t, f])
                                    
                    # update R
                    R_noise_accu = R_noise_accu + lambda_noise[t, f] / phi_noise[t, f] * corre
                    R_noisy_accu = R_noisy_accu + lambda_noisy[t, f] / phi_noisy[t, f] * corre
                          
                # update R
                R_noise[:, :, f] = R_noise_accu / np.sum(lambda_noise[:, f], dtype=np.complex64)    
                R_noisy[:, :, f] = R_noisy_accu / np.sum(lambda_noisy[:, f], dtype=np.complex64) 
    
        # detect noise cluster by entropy        
        for f in range(0, number_of_bins):
            eig_value1 = np.linalg.eigvals(R_noise[:, :, f])
            eig_value2 = np.linalg.eigvals(R_noisy[:, :, f])
            en_noise = np.matmul( - eig_value1.T / np.sum(eig_value1), np.log(eig_value1 / np.sum(eig_value1)))
            en_noisy = np.matmul( - eig_value2.T / np.sum(eig_value2), np.log(eig_value2 / np.sum(eig_value2)))    
            if en_noise < en_noisy:
                Rn = copy.deepcopy(R_noise[:, :, f])
                R_noise[:, :, f] = R_noisy[:, :, f]
                R_noisy[:, :, f] = Rn
        
        R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_n[:, :, f] = R_n[:, :, f] + lambda_noise[t, f] * yyh[:, :, t, f]
            R_n[:, :, f] = R_n[:, :, f] / np.sum(lambda_noise[:, f], dtype=np.complex64)    
        R_x = R_xn - R_n
        return (complex_spectrum, R_x, R_n, lambda_noise, lambda_noisy)  
    
    
    def get_spatial_correlation_matrix_ver2(self, speech_data):
        complex_spectrum, _ = util.get_3dim_spectrum_from_data(speech_data, self.fft_length, self.fft_shift, self.fft_length)
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        
        # CGMM parameters
        lambda_noise = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        lambda_noisy = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        phi_noise = np.ones((number_of_frames, number_of_bins), dtype=np.float64)
        phi_noisy = np.ones((number_of_frames, number_of_bins), dtype=np.float64)
        R_noise = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        R_noisy = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)        
        yyh = np.zeros((number_of_channels, number_of_channels, number_of_frames, number_of_bins), dtype=np.complex64)
        
        # init R_noisy and R_noise
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                h = np.multiply.outer(complex_spectrum[:, t, f], np.conj(complex_spectrum[:, t, f]).T)
                yyh[:, :, t, f] = h
                R_noisy[:, :, f] = R_noisy[:, :, f] + h
            R_noisy[:, :, f] = R_noisy[:, :, f] / number_of_frames
            R_noise[:, :, f] = np.eye(number_of_channels, number_of_channels, dtype=np.complex64)        
        R_xn = copy.deepcopy(R_noisy)                    
        p_noise =  np.ones((number_of_frames, number_of_bins), dtype=np.float64)
        p_noisy =  np.ones((number_of_frames, number_of_bins), dtype=np.float64)
            
        # go EMiteration
        for ite in range(0, self.number_of_EM_iterate):
            print('iter', str(ite + 1) + '/' + str(self.number_of_EM_iterate))
            for f in range(0, number_of_bins):                             
                R_noisy_onbin = copy.deepcopy(R_noisy[:, :, f])
                R_noise_onbin = copy.deepcopy(R_noise[:, :, f])                
                if np.linalg.cond(R_noisy_onbin) < self.condition_number_inv_threshold:
                    R_noisy_onbin = R_noisy_onbin + self.condition_number_inv_threshold * np.eye(number_of_channels) * np.max(np.diag(R_noisy_onbin))                    
                if np.linalg.cond(R_noise_onbin) < self.condition_number_inv_threshold:
                    R_noise_onbin = R_noise_onbin + self.condition_number_inv_threshold * np.eye(number_of_channels) * np.max(np.diag(R_noise_onbin))                
            
                R_noisy_inv = np.linalg.pinv(R_noisy_onbin, rcond=self.scm_inv_threshold)
                R_noise_inv = np.linalg.pinv(R_noise_onbin, rcond=self.scm_inv_threshold)                                            
                R_noisy_accu = np.zeros((number_of_channels, number_of_channels), dtype=np.complex64)
                R_noise_accu = np.zeros((number_of_channels, number_of_channels), dtype=np.complex64)
                
                for t in range(0, number_of_frames):
                    corre = yyh[:, :, t, f]                
                    obs = complex_spectrum[:, t, f]    
                    
                    # update phi (real)
                    phi_noise[t, f] = np.real(np.trace(np.matmul(corre, R_noise_inv), dtype=np.float64) / number_of_channels)
                    phi_noisy[t, f] = np.real(np.trace(np.matmul(corre, R_noisy_inv), dtype=np.float64) / number_of_channels)                    
                    if phi_noise[t, f] == 0:
                        phi_noise[t, f] = self.condition_number_inv_threshold
                    if phi_noisy[t, f] == 0:
                        phi_noisy[t, f] = self.condition_number_inv_threshold
                                            
                    # update p (real)
                    k_noise_1 = np.matmul(np.conj(obs).T , R_noise_inv / phi_noise[t, f])            
                    k_noise = np.matmul(k_noise_1, obs)       
                    tmp_p_noise = np.linalg.det((phi_noise[t, f] * R_noise_onbin).astype(np.float64))
                    p_noise[t, f] = np.real(np.exp( - np.real(k_noise).astype(np.float64)) / (np.pi * tmp_p_noise))                    
                    # avoid nan or inf
                    if np.isnan(p_noise[t, f]) == True or np.isinf(p_noise[t, f]) == True:
                        p_noise[t, f] = np.nan_to_num(p_noise[t, f])                    
                    k_noisy_1 = np.matmul(np.conj(obs).T, R_noisy_inv / phi_noisy[t, f])
                    k_noisy = np.real(np.matmul(k_noisy_1, obs))
                    tmp_p_noisy = np.linalg.det((phi_noisy[t, f] * R_noisy_onbin).astype(np.float64))
                    p_noisy[t, f] = np.real(np.exp( - np.real(k_noisy).astype(np.float64)) / (np.pi * tmp_p_noisy))     
                    # avoid nan or inf
                    if np.isnan(p_noisy[t, f]) == True or np.isinf(p_noisy[t, f]) == True:
                        p_noisy[t, f] = np.nan_to_num(p_noisy[t, f])
            
                    # update lambda
                    lambda_noise[t, f] = p_noise[t, f] / (p_noise[t, f] + p_noisy[t, f])
                    lambda_noisy[t, f] = p_noisy[t, f] / (p_noise[t, f] + p_noisy[t, f])
                                    
                    # update R
                    R_noise_accu = R_noise_accu + lambda_noise[t, f] / phi_noise[t, f] * corre
                    R_noisy_accu = R_noisy_accu + lambda_noisy[t, f] / phi_noisy[t, f] * corre
                          
                # update R
                R_noise[:, :, f] = R_noise_accu / np.sum(lambda_noise[:, f], dtype=np.complex64)    
                R_noisy[:, :, f] = R_noisy_accu / np.sum(lambda_noisy[:, f], dtype=np.complex64) 
    
        # detect noise cluster by entropy        
        for f in range(0, number_of_bins):
            eig_value1 = np.linalg.eigvals(R_noise[:, :, f])
            eig_value2 = np.linalg.eigvals(R_noisy[:, :, f])
            en_noise = np.matmul( - eig_value1.T / np.sum(eig_value1), np.log(eig_value1 / np.sum(eig_value1)))
            en_noisy = np.matmul( - eig_value2.T / np.sum(eig_value2), np.log(eig_value2 / np.sum(eig_value2)))    
            if en_noise < en_noisy:
                Rn = copy.deepcopy(R_noise[:, :, f])
                R_noise[:, :, f] = R_noisy[:, :, f]
                R_noisy[:, :, f] = Rn
        
        R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        R_x = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_n[:, :, f] = R_n[:, :, f] + lambda_noise[t, f] * yyh[:, :, t, f]
            R_n[:, :, f] = R_n[:, :, f] / np.sum(lambda_noise[:, f], dtype=np.complex64)  
            
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_x[:, :, f] = R_x[:, :, f] + lambda_noisy[t, f] * yyh[:, :, t, f]
            R_x[:, :, f] = R_x[:, :, f] / np.sum(lambda_noisy[:, f], dtype=np.complex64)    
            
        #R_x = R_xn - R_n
        return (complex_spectrum, R_x, R_n, lambda_noise, lambda_noisy)  


    def get_spatial_correlation_matrix_from_mask(self, speech_data, speech_mask, noise_mask=np.array([None])):      
        if noise_mask.any() == None:
            print('make_noise_mask')
            noise_mask = (1 - speech_mask)+0.01
        else:
            noise_mask = noise_mask.T
        print(np.shape(speech_mask), np.shape(noise_mask))
        complex_spectrum, _ = util.get_3dim_spectrum_from_data(speech_data, self.fft_length, self.fft_shift, self.fft_length)
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)       
        # safe guard for difference size between speakerbeam's mask and complex spectrum
        _, number_of_frames_on_speakerbeam_mask = np.shape(noise_mask)
        if number_of_frames != number_of_frames_on_speakerbeam_mask:
            maximum_number_of_frames = np.min([number_of_frames, number_of_frames_on_speakerbeam_mask])
            complex_spectrum = complex_spectrum[:, 0:maximum_number_of_frames, :]
            speech_mask = speech_mask[:, 0:maximum_number_of_frames]
            noise_mask = noise_mask[:, 0:maximum_number_of_frames]
            number_of_frames = maximum_number_of_frames       
            print(maximum_number_of_frames)
        noise_mask = noise_mask.T
        speech_mask = speech_mask.T
        yyh = np.zeros((number_of_channels, number_of_channels, number_of_frames, number_of_bins), dtype=np.complex64)
        R_xn = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)                
        # init R_noisy and R_noise
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                h = np.multiply.outer(complex_spectrum[:, t, f], np.conj(complex_spectrum[:, t, f]).T)
                yyh[:, :, t, f] = h
                R_xn[:, :, f] = R_xn[:, :, f] + h
        R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_n[:, :, f] = R_n[:, :, f] + noise_mask[t, f] * yyh[:, :, t, f]
            R_n[:, :, f] = R_n[:, :, f] / np.sum(noise_mask[:, f], dtype=np.complex64)    
        R_x = R_xn - R_n
        return (complex_spectrum, R_x, R_n, noise_mask, speech_mask)  

    def get_spatial_correlation_matrix_from_mask_for_LSTM(self, speech_data, speech_mask, noise_mask=np.array([None]), less_frame=3):      
        """
        if noise_mask.any() == None:
            print('make_noise_mask')
            noise_mask = (1 - speech_mask)+0.01
        else:
            noise_mask = noise_mask.T
        """
        #print(np.shape(speech_mask), np.shape(noise_mask))
        complex_spectrum, _ = util.get_3dim_spectrum_from_data(speech_data, self.fft_length, self.fft_shift, self.fft_length)
        tmp_complex_spectrum  = copy.deepcopy(complex_spectrum)
        # safe guard for difference size between speakerbeam's mask and complex spectrum
        
        # ad-hock selection 5/14
        complex_spectrum = complex_spectrum[:, less_frame:-(less_frame + 1), :]
        #speech_mask = speech_mask[:, less_frame:-(less_frame + 1)]
        #noise_mask = noise_mask[:, less_frame:-(less_frame + 1)]
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)               
        _, number_of_frames_on_speakerbeam_mask = np.shape(noise_mask)
        
        if number_of_frames != number_of_frames_on_speakerbeam_mask:
            maximum_number_of_frames = np.min([number_of_frames, number_of_frames_on_speakerbeam_mask])
            complex_spectrum = complex_spectrum[:, 0:maximum_number_of_frames, :]
            speech_mask = speech_mask[:, 0:maximum_number_of_frames]
            noise_mask = noise_mask[:, 0:maximum_number_of_frames]
            number_of_frames = maximum_number_of_frames       
        noise_mask = np.fliplr(noise_mask.T)
        speech_mask = np.fliplr(speech_mask.T)
        """
        pl.figure()
        
        pl.imshow(noise_mask, aspect='auto')
        pl.title('n_mask_median')
        pl.figure()
        pl.imshow(speech_mask, aspect='auto')
        pl.title('s_mask_median')
        pl.show()
        """
        yyh = np.zeros((number_of_channels, number_of_channels, number_of_frames, number_of_bins), dtype=np.complex64)
        R_xn = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)                
        # init R_noisy and R_noise
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                h = np.multiply.outer(complex_spectrum[:, t, f], np.conj(complex_spectrum[:, t, f]).T)
                yyh[:, :, t, f] = h
                R_xn[:, :, f] = R_xn[:, :, f] + h
        R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_n[:, :, f] = R_n[:, :, f] + noise_mask[t, f] * yyh[:, :, t, f]
            R_n[:, :, f] = R_n[:, :, f] / np.sum(noise_mask[:, f], dtype=np.complex64)    
        R_x = R_xn - R_n
        return (tmp_complex_spectrum, R_x, R_n, noise_mask, speech_mask)  

    
    def get_spatial_correlation_matrix_from_mask_for_LSTM_ver2(self, speech_data, speech_mask, noise_mask=np.array([None]), less_frame=10):      
        """
        if noise_mask.any() == None:
            print('make_noise_mask')
            noise_mask = (1 - speech_mask)+0.01
        else:
            noise_mask = noise_mask.T
            
        return noise/speech SCM respectivily
        """
        #print(np.shape(speech_mask), np.shape(noise_mask))
        complex_spectrum, _ = util.get_3dim_spectrum_from_data(speech_data, self.fft_length, self.fft_shift, self.fft_length)
        tmp_complex_spectrum  = copy.deepcopy(complex_spectrum)
        # safe guard for difference size between speakerbeam's mask and complex spectrum
        
        # ad-hock selection 5/14
        complex_spectrum = complex_spectrum[:, less_frame:-(less_frame + 1), :]
        #speech_mask = speech_mask[:, less_frame:-(less_frame + 1)]
        #noise_mask = noise_mask[:, less_frame:-(less_frame + 1)]
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)               
        _, number_of_frames_on_speakerbeam_mask = np.shape(noise_mask)
        
        if number_of_frames != number_of_frames_on_speakerbeam_mask:
            maximum_number_of_frames = np.min([number_of_frames, number_of_frames_on_speakerbeam_mask])
            complex_spectrum = complex_spectrum[:, 0:maximum_number_of_frames, :]
            speech_mask = speech_mask[:, 0:maximum_number_of_frames]
            noise_mask = noise_mask[:, 0:maximum_number_of_frames]
            number_of_frames = maximum_number_of_frames       
        noise_mask = np.fliplr(noise_mask.T)
        speech_mask = np.fliplr(speech_mask.T)
        """
        pl.figure()
        
        pl.imshow(noise_mask, aspect='auto')
        pl.title('n_mask_median')
        pl.figure()
        pl.imshow(speech_mask, aspect='auto')
        pl.title('s_mask_median')
        pl.show()
        """
        yyh = np.zeros((number_of_channels, number_of_channels, number_of_frames, number_of_bins), dtype=np.complex64)
        R_xn = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)                
        # init R_noisy and R_noise
        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                h = np.multiply.outer(complex_spectrum[:, t, f], np.conj(complex_spectrum[:, t, f]).T)
                yyh[:, :, t, f] = h
                R_xn[:, :, f] = R_xn[:, :, f] + h
        R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        R_x = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)

        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_n[:, :, f] = R_n[:, :, f] + noise_mask[t, f] * yyh[:, :, t, f]
            R_n[:, :, f] = R_n[:, :, f] / np.sum(noise_mask[:, f], dtype=np.complex64)    

        for f in range(0, number_of_bins):
            for t in range(0, number_of_frames):
                R_x[:, :, f] = R_x[:, :, f] + speech_mask[t, f] * yyh[:, :, t, f]
            R_x[:, :, f] = R_x[:, :, f] / np.sum(speech_mask[:, f], dtype=np.complex64)    
            
        #R_x = R_x - R_n
        return (tmp_complex_spectrum, R_x, R_n, noise_mask, speech_mask)  

                     
    def get_mvdr_beamformer(self, R_x, R_n):
        number_of_channels, _, number_of_bins = np.shape(R_x)
        beamformer = np.ones((number_of_channels, number_of_bins), dtype=np.complex64)        
        for f in range(0, number_of_bins):
            _, eigen_vector = np.linalg.eig(R_x[:, :, f])
            steering_vector = eigen_vector[:, 0]
            Rn_inv = np.linalg.pinv(R_n[:, :, f], rcond=self.beamformer_inv_threshold)
            w1 = np.matmul(Rn_inv, steering_vector)
            w2 = np.matmul(np.conjugate(steering_vector).T, Rn_inv)
            w2 = np.matmul(w2, steering_vector)
            w2 = np.reshape(w2, [1, 1])
            w = w1 / w2
            w = np.reshape(w, number_of_channels)
            beamformer[:, f] = w
        return (beamformer, steering_vector)
    
    def get_mvdr_beamformer_by_maxsnr(self, R_x, R_n):
        '''
        Improved MVDR beamforming using single-channel mask
                prediction networks [Erdogan, 2016]
        '''
        
        number_of_channels, _, number_of_bins = np.shape(R_x)
        # beamformer >> (selectablebeam, number_of_channels, number_of_bins)
        beamformer = np.ones((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)  
        # all channles beamformer
        selected_SNR = np.zeros(number_of_channels, dtype=np.float32)
        for c in range(0, number_of_channels):
            r = np.zeros(number_of_channels, dtype=np.complex64)
            r[c] = 1
            for f in range(0, number_of_bins):
                Rn_inv = np.linalg.pinv(R_n[:, :, f], rcond=self.beamformer_inv_threshold)
                w1_1 = np.matmul(Rn_inv, R_x[:, :, f])
                w1 = np.matmul(w1_1, r)
                # normalize factor
                w2 = np.trace(w1_1)
                w2 = np.reshape(w2, [1, 1])
                w = w1 / w2
                w = np.reshape(w, number_of_channels)
                beamformer[c, :, f] = w
            w1_sum = 0
            w2_sum = 0
            for f2 in range(0, number_of_bins):
                snr_post_w1 = np.matmul(np.conjugate(beamformer[c, :, f2]).T, R_x[:, :, f2])
                snr_post_w1 = np.matmul(snr_post_w1, beamformer[c, :, f2])
                snr_post_w2 = np.matmul(np.conjugate(beamformer[c, :, f2]).T, R_n[:, :, f2])
                snr_post_w2 = np.matmul(snr_post_w2, beamformer[c, :, f2])
                w1_sum = w1_sum + snr_post_w1
                w2_sum = w2_sum + snr_post_w2
            selected_SNR[c] = np.float32(w1_sum) / np.float32(w2_sum)
        print('snr', selected_SNR)
        max_index = np.argmax(selected_SNR)                       
        return beamformer[max_index, :, :]
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return util.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)
