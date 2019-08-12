# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:28:39 2019

@author: a-kojima
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:29:57 2019

@author: a-kojima
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as pl
from scipy import signal as sg
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh

from beamformer import complexGMM_mvdr as cgmm
from beamformer import util
from maskestimator import model, shaper, feature




def get_stack_speech(speech1, speech2):
    return np.concatenate((speech1, speech2))

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
RECURRENT_INIT = 0.00001
SPEECH_PATH = r'./dataset/adaptation_data/speaker1_2/251-137823-0023.flac'
WEIGHT_PATH_ORI = r'./model/194sequence_false_e1.hdf5' #194
WEIGHT_PATH_SP1 = r'./model/speaker_1.hdf5'
WEIGHT_PATH_SP2 = r'./model/speaker_2.hdf5'
NUMBER_OF_STACK = LEFT_CONTEXT + RIGHT_CONTEXT + 1

#==========================================
# get model
#==========================================
mask_estimator_generator1 = model.NeuralMaskEstimation(TRUNCATE_GRAD,
                                            NUMBER_OF_STACK,
                                            0.1, 
                                            FFTL // 2 + 1,
                                            recurrent_init=RECURRENT_INIT)
mask_estimator1 = mask_estimator_generator1.get_model(is_stateful=True, is_show_detail=False, is_adapt=False,)
mask_estimator1 = mask_estimator_generator1.load_weight_param(mask_estimator1, WEIGHT_PATH_SP1)

mask_estimator_generator2 = model.NeuralMaskEstimation(TRUNCATE_GRAD,
                                            NUMBER_OF_STACK,
                                            0.1, 
                                            FFTL // 2 + 1,
                                            recurrent_init=RECURRENT_INIT)
mask_estimator2 = mask_estimator_generator2.get_model(is_stateful=True, is_show_detail=False, is_adapt=False)
mask_estimator2 = mask_estimator_generator2.load_weight_param(mask_estimator2, WEIGHT_PATH_SP2)

mask_estimator_generator_ori = model.NeuralMaskEstimation(TRUNCATE_GRAD,
                                            NUMBER_OF_STACK,
                                            0.1, 
                                            FFTL // 2 + 1,
                                            recurrent_init=RECURRENT_INIT)
mask_estimator_ori = mask_estimator_generator_ori.get_model(is_stateful=True, is_show_detail=False, is_adapt=False)

mask_estimator_ori = mask_estimator_generator_ori.load_weight_param(mask_estimator_ori, WEIGHT_PATH_ORI)


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
speech = sf.read(SPEECH_PATH)[0]


noisy_spectrogram = feature_extractor.get_feature(speech)
noisy_spectrogram = np.flipud(noisy_spectrogram)    
noisy_spectrogram = feature_extractor.apply_cmvn(noisy_spectrogram)
    
features = data_shaper.convert_for_predict(noisy_spectrogram)
features = np.array(features)
padding_feature, original_batch_size = data_shaper.get_padding_features(features)

mask_estimator1.reset_states()    
sp_mask1, n_mask1 = mask_estimator1.predict_on_batch(padding_feature)
sp_mask1 = sp_mask1[:original_batch_size, :]
n_mask1 = n_mask1[:original_batch_size, :]

mask_estimator2.reset_states()    
sp_mask2, n_mask2 = mask_estimator2.predict_on_batch(padding_feature)
sp_mask2 = sp_mask2[:original_batch_size, :]
n_mask2 = n_mask2[:original_batch_size, :]


mask_estimator_ori.reset_states()    
sp_mask_ori, n_mask_ori = mask_estimator_ori.predict_on_batch(padding_feature)
sp_mask_ori = sp_mask_ori[:original_batch_size, :]
n_mask_ori = n_mask_ori[:original_batch_size, :]


pl.figure(),
pl.subplot(2, 1, 1)
pl.imshow(n_mask1.T, aspect='auto')
pl.title('sp1_original')
pl.subplot(2, 1, 2)
pl.imshow(sp_mask1.T, aspect='auto')

pl.figure(),
pl.subplot(2, 1, 1)
pl.imshow(n_mask2.T, aspect='auto')
pl.title('sp2_original')
pl.subplot(2, 1, 2)
pl.imshow(sp_mask2.T, aspect='auto')


pl.figure(),
pl.subplot(2, 1, 1)
pl.imshow(n_mask_ori.T, aspect='auto')
pl.title('original_model_predict')
pl.subplot(2, 1, 2)
pl.imshow(sp_mask_ori.T, aspect='auto')

# ====================================
# subract and calculate final mask
# ====================================
sub_mask_sp_mask = sp_mask1 - sp_mask2
sub_mask_sp_mask[sub_mask_sp_mask<=0] = 0

pl.figure(),
pl.subplot(2, 1, 1)
pl.imshow(1 - sub_mask_sp_mask.T, aspect='auto')
pl.title('speaker1-aware mask')
pl.subplot(2, 1, 2)
pl.imshow(sub_mask_sp_mask.T, aspect='auto')

pl.show()

