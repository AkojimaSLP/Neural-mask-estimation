# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:29:57 2019

@author: a-kojima
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as pl

from beamformer import complexGMM_mvdr as cgmm
from beamformer import complexGMM_mvdr_snr_selective as cgmm_snr
from maskestimator import model, shaper, feature

#==========================================
# ANALYSIS PARAMETERS
#==========================================
SAMPLING_FREQUENCY = 16000
FFTL = 1024
SHIFT = 256

#==========================================
# ESURAL MASL ESTIMATOR PARAMETERS
#==========================================
LEFT_CONTEXT = 0
RIGHT_CONTEXT = 0
NUMBER_OF_SKIP_FRAME = 0  

#==========================================
# ESURAL MASL ESTIMATOR TRAINNING PARAMERTERS
#==========================================
TRUNCATE_GRAD = 7
IS_DEBUG_SHOW_PREDICT_MASK = True

NOISY_SPEECH_PATH = r'./dataset/data_for_beamforming/F02_011C021A_BUS.CH{}.wav'
CHANNEL_INDEX = [1, 2, 3, 4, 5, 6]
WEIGHT_PATH = r'./model/194sequence_false_e1.hdf5'

NUMBER_OF_STACK = LEFT_CONTEXT + RIGHT_CONTEXT + 1

OPERATION = 'median'
RECURRENT_CELL_INIT = 0.00001 #0.04

MAX_SEQUENCE = 5000

#==========================================
# get model
#==========================================
mask_estimator_generator = model.NeuralMaskEstimation(TRUNCATE_GRAD,
                                            NUMBER_OF_STACK,
                                            0.1, 
                                            FFTL // 2 + 1,
                                            recurrent_init=RECURRENT_CELL_INIT)

mask_estimator = mask_estimator_generator.get_model(is_stateful=True, is_show_detail=True, is_adapt=False)

mask_estimator = mask_estimator_generator.load_weight_param(mask_estimator, WEIGHT_PATH)
#==========================================
# predicting data shaper
#==========================================
data_shaper = shaper.Shape_data(LEFT_CONTEXT,
                  RIGHT_CONTEXT,
                  TRUNCATE_GRAD,
                  NUMBER_OF_SKIP_FRAME )

#==========================================
# get features
#==========================================
feature_extractor = feature.Feature(SAMPLING_FREQUENCY, FFTL, SHIFT)

for ii in range(0, len(CHANNEL_INDEX)):
    speech = sf.read(NOISY_SPEECH_PATH.replace('{}', str(CHANNEL_INDEX[ii])))[0]
    
    noisy_spectrogram = feature_extractor.get_feature(speech)
    noisy_spectrogram = (np.flipud(noisy_spectrogram))    
    noisy_spectrogram = feature_extractor.apply_cmvn(noisy_spectrogram)
        
    features = data_shaper.convert_for_predict(noisy_spectrogram)
    features = np.array(features)

    mask_estimator.reset_states()    

    padding_feature, original_batch_size = data_shaper.get_padding_features(features)
    sp_mask, n_mask = mask_estimator.predict(padding_feature, batch_size=MAX_SEQUENCE)
    sp_mask = sp_mask[:original_batch_size, :]
    n_mask = n_mask[:original_batch_size, :]
    
    if IS_DEBUG_SHOW_PREDICT_MASK == True:
        pl.subplot(len(CHANNEL_INDEX), 2, ((ii + 1) * 2) - 1)
        pl.imshow(((n_mask).T), aspect='auto')    
        pl.subplot(len(CHANNEL_INDEX), 2, ((ii + 1) * 2))
        pl.imshow(((sp_mask).T), aspect='auto')
    
    
    if ii == 0:
        aa,bb = np.shape(n_mask)
        n_median = np.zeros((aa,bb,len(CHANNEL_INDEX)))
        sp_median = np.zeros((aa,bb,len(CHANNEL_INDEX)))
        
        n_median[:,:,ii] = n_mask
        sp_median[:,:,ii] = sp_mask
        dump_speech = np.zeros((len(speech), len(CHANNEL_INDEX)))
        dump_speech[:, ii] = speech           
    else:
        n_median[:,:,ii] = n_mask
        sp_median[:,:,ii] = sp_mask
        dump_speech[:, ii] = speech        
        
if OPERATION == 'median':
    n_median_s = np.median(n_median, axis=2)
    sp_median_s = np.median(sp_median, axis=2)
else:
    n_median_s = np.mean(n_median, axis=2)
    sp_median_s = np.mean(sp_median, axis=2)


if IS_DEBUG_SHOW_PREDICT_MASK == True:
    pl.figure()
    pl.subplot(3,1,1)
    pl.imshow((np.log10(noisy_spectrogram[:, TRUNCATE_GRAD // 2:- TRUNCATE_GRAD // 2] ** 2) * 10), aspect='auto')
    pl.subplot(3,1,2)
    pl.imshow(((n_median_s.T)),  aspect = "auto")
    pl.title('noise mask')
    pl.subplot(3,1,3)
    pl.imshow(((sp_median_s.T)),  aspect = "auto")
    pl.title('speech mask')
    pl.show()

#==========================================
# beamforming
#==========================================

# sinple MVDR    
cgmm_bf = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFTL, SHIFT, 10, 10)
tmp_complex_spectrum, R_x, R_n, tt, nn = cgmm_bf.get_spatial_correlation_matrix_from_mask_for_LSTM(dump_speech,
                                                                                                 speech_mask=sp_median_s.T,
                                                                       noise_mask=n_median_s.T,
                                                                                           less_frame=3)
beamformer, steering_vector = cgmm_bf.get_mvdr_beamformer(R_x, R_n)
enhan_speech = cgmm_bf.apply_beamformer(beamformer, tmp_complex_spectrum)

# reference mic selection MVDR
cgmm_bf_snr = cgmm_snr.complexGMM_mvdr(SAMPLING_FREQUENCY, FFTL, SHIFT, 10, 10)

tmp_complex_spectrum, R_x, R_n, tt, nn = cgmm_bf_snr.get_spatial_correlation_matrix_from_mask_for_LSTM(dump_speech,
                                                                                                 speech_mask=sp_median_s.T,
                                                                       noise_mask=n_median_s.T,
                                                                                           less_frame=3)

selected_beamformer = cgmm_bf_snr.get_mvdr_beamformer_by_maxsnr(R_x, R_n)
enhan_speech2 = cgmm_bf_snr.apply_beamformer(selected_beamformer, tmp_complex_spectrum)

enhan_speech = enhan_speech / np.max(np.abs(enhan_speech)) * 0.75
enhan_speech2 = enhan_speech2 / np.max(np.abs(enhan_speech2)) * 0.75
sf.write('./result/enhacement_all_channels.wav', enhan_speech, 16000)
sf.write('./result/enhacement_snr_select.wav', enhan_speech2, 16000)
