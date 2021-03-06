# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:29:31 2019

@author: a-kojima

Neural mask estimation for MVDR

this script suppots on-the-fly training for data-augmentation efficiently

"""
import numpy as np
import glob
from scipy import stats
import random
import soundfile as sf
import matplotlib.pyplot as pl
import sys

from maskestimator import model, shaper, feature, augment, util
from tensorflow.keras.callbacks import TensorBoard


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
# NEURAL MASL ESTIMATOR TRAINNING PARAMERTERS
#==========================================
EPOCHS = 1
NUMBER_OF_UTTERANCE = 1 #15
TRUNCATE_GRAD = 7
LR = 0.001 
SPEECH_DIRECTORY = r'./dataset/train/speech/*'
NOISE_DIRECTORY = r'./dataset/train/noise/*'
IS_DEBUG_SHOW_MASK_AND_SYNTHESIS = False
MODEL_NAME_PREFIX = r'./model/neaural_mask_estimator{}.hdf5'
LOSS_NAME_PREFIX = r'./model/neaural_mask_estimator.npy'
LOG_PATH = r'./tflog/' 
RECURRENT_INIT = 0.04
VALIDATION_SPEC = r'./validation_features/val_spec.npy'
VALIDATION_SPEECH_MASK = r'./validation_features/noise_mask.npy'
VALIDATION_NOISE_MASK = r'./validation_features/speech_mask.npy'


NUMBER_OF_STACK = LEFT_CONTEXT + RIGHT_CONTEXT + 1

#==========================================
# augmentation parameters
#==========================================
'''
snr: [SNR20, SNR15, SNR10, SNR5, SNR0]
prob.: [0.2,... 0.2]
'''
SNR_generator = stats.rv_discrete(values=(np.array([0, 1, 2, 3, 4]),
                                          (0.2, 0.2, 0.2, 0.2, 0.2)))
noise_list = glob.glob(NOISE_DIRECTORY)
RIR_CONVOLVE_CHANCE_RATE = 0.25 # ex., 0.5 means convolution speech with RIR with 50% chanve rate

#==========================================
# prepare speech and noise file list
#==========================================
speech_list = glob.glob(SPEECH_DIRECTORY)
noise_list = glob.glob(NOISE_DIRECTORY)

#==========================================
# get model
#==========================================
mask_estimator_generator = model.NeuralMaskEstimation(TRUNCATE_GRAD,
                                            NUMBER_OF_STACK,
                                            LR, 
                                            FFTL // 2 + 1,
                                            recurrent_init=RECURRENT_INIT)
mask_estimator = mask_estimator_generator.get_model(is_stateful=False,
                                                    is_show_detail=True,
                                                    is_adapt=False)

#==========================================
# training data shaper
#==========================================
data_shaper = shaper.Shape_data(LEFT_CONTEXT,
                  RIGHT_CONTEXT,
                  TRUNCATE_GRAD,
                  NUMBER_OF_SKIP_FRAME )

#==========================================
# set tensorboard
#==========================================
callback = TensorBoard(LOG_PATH)
callback.set_model(mask_estimator)

#==========================================
# get features
#==========================================
feature_extractor = feature.Feature(SAMPLING_FREQUENCY, FFTL, SHIFT)
noise_generator = augment.Generate_random_noise(noise_list, SAMPLING_FREQUENCY)
reverbarent_generator = augment.RIR_convolve(SAMPLING_FREQUENCY)

#==========================================
# go training
#==========================================
utterance_count = 0
TRIM = np.int(0.05 * SAMPLING_FREQUENCY) # beginning and ending of uttearnce is not used for training
val_loss = np.array([])
test_loss = np.array([])
validate_features = np.load(VALIDATION_SPEC)
validate_label_sp = np.load(VALIDATION_SPEECH_MASK)
validate_label_n = np.load(VALIDATION_NOISE_MASK)
freq_grid = np.linspace(0, SAMPLING_FREQUENCY, FFTL)[0:FFTL // 2 + 1]
bin_index = np.argmin(np.abs(freq_grid - 2000))


for i in range(0, EPOCHS):
    speech_list_shuffle = random.sample(speech_list, len(speech_list))
    
    # go NN parameters optimizer
    while True:
        feature_stack = []
        label_stack_sp = []
        label_stack_n = []        
        if len(speech_list_shuffle) < NUMBER_OF_UTTERANCE:
            break        
        
        # dumping frame until searching # of utterances
        while True:
            # all utterance is used for training
            if len(speech_list_shuffle) <= NUMBER_OF_UTTERANCE:
                break
            
            index = np.random.randint(0, len(speech_list_shuffle) - 1, 1)[0]            
            audio_path = speech_list_shuffle[index]
            speech_list_shuffle.pop(index) # remove uterance chosen yet
            
            
            speech = sf.read(audio_path, dtype='float32')[0]    
            speech = speech[TRIM:-TRIM]
            
            if  len(speech) != 0:
                speech = feature_extractor.add_white_noise(speech)       
                if IS_DEBUG_SHOW_MASK_AND_SYNTHESIS == True:
                    sf.write('./result/speech_clean.wav', speech , 16000)
                SNR_index = SNR_generator.rvs(size=1)[0]
                noise = noise_generator.get_noise(len(speech))
                noise = feature_extractor.add_white_noise(noise)
                
                if RIR_CONVOLVE_CHANCE_RATE != 0:
                    # convolve RIR
                    if np.random.randint(0, 1 // RIR_CONVOLVE_CHANCE_RATE, 1)[0] == 1:
                        speech, noise = reverbarent_generator.get_reverbant_speech(speech, noise) 
                        
                snr_adjuster = augment.SNR_adjusting(speech, noise)                        
                if SNR_index == 0:            
                    SNR = 20
                elif SNR_index == 1:
                    SNR = 15
                elif SNR_index == 2:
                    SNR = 10
                elif SNR_index == 3:
                    SNR = 5
                elif SNR_index == 4:
                    SNR = 0   
                speech, noise = snr_adjuster.add_speech_to_noise(SNR)
                speech, noise = snr_adjuster.avoid_clipping(speech, noise)  

                # if get mask after SNR adjusting
                speech_spectrogram = feature_extractor.get_feature(speech)
                noise_spectrogram = feature_extractor.get_feature(noise)        
                freq_grid = np.linspace(0, SAMPLING_FREQUENCY, FFTL)[0:FFTL // 2 + 1]
                bin_index = np.argmin(np.abs(freq_grid - 2000))
                speech_mask, noise_mask = feature_extractor.get_ideal_binary_mask_herman(speech_spectrogram,
                                                                                         noise_spectrogram,
                                                                                         threshold_bin=bin_index,
                                                                                         theta_sp_low=10**(-4),
                                                                                         theta_sp_high=10**(-5),
                                                                                         theta_n_low=10**(-5),#-0.01
                                                                                         theta_n_high=10**(-5)) #-0.02
                                                               
                noisy_spectrogram = (speech_spectrogram + noise_spectrogram)                
                noisy_spectrogram = (np.flipud(noisy_spectrogram))
                speech_mask = np.flipud(speech_mask)
                noise_mask = np.flipud(noise_mask)
                
                noisy_spectrogram = feature_extractor.apply_cmvn(noisy_spectrogram)
                noisy_spectrogram = noisy_spectrogram + np.random.normal(loc=0, scale=0.0001, size=np.shape(noisy_spectrogram))
                features, label_sp, label_n = data_shaper.convert_for_train(noisy_spectrogram, speech_mask, noise_mask)
                                
                if len(features) != 0:
                    features = np.array(features)              
                    label_sp = np.array(label_sp)
                    label_n = np.array(label_n)
                    if IS_DEBUG_SHOW_MASK_AND_SYNTHESIS == True:
                        sf.write('./result/speech_noisy.wav', speech + noise, 16000)                        
                        pl.figure(),
                        pl.imshow(noise_mask, aspect='auto', extent=[0, np.shape(noise_mask)[1], 0, 8000,])
                        pl.title('noise mask')
                        pl.figure(),
                        pl.imshow(speech_mask, aspect='auto',extent=[0, np.shape(noise_mask)[1], 0, 8000])
                        pl.title('sp mask')
                        pl.figure()
                        pl.imshow(noisy_spectrogram, aspect='auto')
                        pl.show()
                        sys.exit()                    
                    feature_stack.extend(features)         
                    label_stack_sp.extend(label_sp)
                    label_stack_n.extend(label_n)
                    utterance_count = utterance_count + 1                    
                if utterance_count == NUMBER_OF_UTTERANCE:
                    break          
        
        train_features = np.array(feature_stack)
        train_label_sp = np.array(label_stack_sp)
        train_label_n = np.array(label_stack_n)

        
        if np.shape(train_features)[0] != 0:
            shuffle_index = random.sample(range(0, np.shape(train_features)[0]), np.shape(train_features)[0])
            history = mask_estimator.train_on_batch(x=train_features[shuffle_index, :, :], 
                      y=[train_label_sp[shuffle_index, :], train_label_n[shuffle_index, :]])
            print('train_loss:', history)
            print('epoch:', i)
        else:
            break
        
        # reset 
        feature_stack = []
        label_stack = []
        utterance_count = 0

    # evaluate validation data    
    val_loss_r = mask_estimator.evaluate(x=validate_features,
                   y=[validate_label_sp, validate_label_n],
                   verbose=0)
    print('val_loss:', val_loss_r)
    util.write_log(callback, ['val_loss', 'speech_loss', 'noise_loss', 'accu_speech', 'accu_noise'], val_loss_r, i)
        
    val_loss = np.append(val_loss, val_loss_r)
    np.save(LOSS_NAME_PREFIX, val_loss)
    mask_estimator.save_weights(MODEL_NAME_PREFIX.replace('{}', str(i)))

