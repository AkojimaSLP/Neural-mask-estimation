# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:29:57 2019

@author: a-kojima
"""

import os
import shutil

from maskestimator import model, shaper, adapt_model

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
TRUNCATE_GRAD = 7

#==========================================
# ESURAL MASL ESTIMATOR TRAINNING PARAMERTERS
#==========================================
WEIGHT_PATH = r'./model/194sequence_false_e1.hdf5'
ADAPT_LR = 0.001

TARGET_SPEAKER_LIST = './sp2_list.txt'
NON_TARGET_SPEAKER_LIST = './sp1_list.txt'
SAVE_MODEL_NAME = r'./model/speaker_2.hdf5'
ADAPT_LOC = r'./adapt_data' # place to output numpy features for adaptation
RECURRENT_INIT_CELL = 0.00001



NUMBER_OF_STACK = LEFT_CONTEXT + RIGHT_CONTEXT + 1

#==========================================
# get model
#==========================================
mask_estimator_generator = model.NeuralMaskEstimation(TRUNCATE_GRAD,
                                            NUMBER_OF_STACK,
                                            ADAPT_LR, 
                                            FFTL // 2 + 1,
                                            recurrent_init=RECURRENT_INIT_CELL)

mask_estimator = mask_estimator_generator.get_model(is_stateful=True,
                                                    is_show_detail=True,
                                                    is_adapt=False)

mask_estimator = mask_estimator_generator.load_weight_param(mask_estimator, WEIGHT_PATH)

#==========================================
# predicting data shaper
#==========================================
data_shaper = shaper.Shape_data(LEFT_CONTEXT,
                  RIGHT_CONTEXT,
                  TRUNCATE_GRAD,
                  NUMBER_OF_SKIP_FRAME )

#==========================================
# adaptation
#==========================================
model_adapter = adapt_model.adapt_model(WEIGHT_PATH,
                        TRUNCATE_GRAD, 
                        NUMBER_OF_STACK,
                        ADAPT_LR,
                        spec_dim=FFTL // 2 + 1,
                        sampling_frequency=SAMPLING_FREQUENCY,
                        fftl=FFTL,
                        shift=SHIFT,
                        left_context=LEFT_CONTEXT,
                        right_contect=RIGHT_CONTEXT,
                        number_of_skip_frame=NUMBER_OF_SKIP_FRAME,                        
                        adapt_data_location=ADAPT_LOC)

#==========================================
# create data for adaptation
#==========================================
if os.path.exists(ADAPT_LOC):
    shutil.rmtree(ADAPT_LOC)
os.makedirs(ADAPT_LOC)

# target speaker
model_adapter.create_data_for_adaptation(True, TARGET_SPEAKER_LIST)                        
# non target speaker                        
model_adapter.create_data_for_adaptation(False, NON_TARGET_SPEAKER_LIST)

#==========================================
# adaptation
#==========================================
model_adapter.save_adapt_model(SAVE_MODEL_NAME)
