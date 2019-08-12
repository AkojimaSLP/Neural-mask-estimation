# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:25:12 2019

@author: a-kojima
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

DROPOUT = 0.5
MAX_SEQUENCE = 5000

class NeuralMaskEstimation:
        
    def __init__(self,
                 truncate_grad,
                 number_of_stack,
                 lr,
                 spec_dim,
                 ff_dropout=0,
                 recurrent_dropout=0,
                 recurrent_init=0.04):
        self.truncate_grad = truncate_grad
        self.number_of_stack = number_of_stack
        self.lr = lr
        self.spec_dim = spec_dim # 513
        self.ff_dropout = ff_dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_init=recurrent_init
    
    def get_model(self,
                  is_stateful=True,
                  is_show_detail=True, 
                  is_adapt=False):
        if is_stateful == True:
            input_sequence = layers.Input(shape=(self.truncate_grad, self.spec_dim * self.number_of_stack), batch_size=MAX_SEQUENCE) # time step * feature_size
        else:
            input_sequence = layers.Input(shape=(self.truncate_grad, self.spec_dim * self.number_of_stack)) # time step * feature_size
        LSTM_layer = (LSTM(self.spec_dim,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       stateful=is_stateful,
                       dropout=self.ff_dropout,
                       recurrent_dropout=self.recurrent_dropout,    
                       go_backwards=False,
                       unroll=True,
                       recurrent_initializer=initializers.RandomUniform(minval=-self.recurrent_init, maxval=self.recurrent_init),  #0.04
                       name='lstm'
                       ))(input_sequence)
        DROPOUT1 = Dropout(DROPOUT, name='dropout1')(LSTM_layer)        
        FC1 = Dense(self.spec_dim,activation='relu', name='fc1')(DROPOUT1)
        FC2 = Dropout(DROPOUT, name='dropout2')(FC1)
        FC3 = Dense(self.spec_dim,activation='relu', name='fc2')(FC2)
        DROPOUT2 = Dropout(DROPOUT, name='dropout3')(FC3)
        OUTPUT1 = Dense(self.spec_dim,
                    activation='sigmoid',
                    name='speech_mask')(DROPOUT2)      

        OUTPUT2 = Dense(513,
                    activation='sigmoid',
                    name='noise_mask')(DROPOUT2)                    
        model = Model(inputs=[input_sequence], outputs=[OUTPUT1, OUTPUT2])
        
        if is_adapt == False:
            model.compile(
                    loss={'speech_mask':'binary_crossentropy', 'noise_mask':'binary_crossentropy'},
                          metrics=['acc'],
                          sample_weight_mode="None",
                          loss_weights={'speech_mask':1, 'noise_mask':1},
                          optimizer=optimizers.RMSprop(lr=self.lr, decay=1e-6, epsilon=1e-06, clipnorm=1.0))
        else:
            model.compile(
                    loss={'speech_mask':'binary_crossentropy', 'noise_mask':'binary_crossentropy'},
                          metrics=['acc'],
                          sample_weight_mode="None",
                          loss_weights={'speech_mask':1, 'noise_mask':0},
                          #optimizer=optimizers.RMSprop(lr=self.lr, decay=1e-6, epsilon=1e-06))
                          optimizer=optimizers.RMSprop(lr=self.lr, clipnorm=1.0))
            
        if is_show_detail == True:
            model.summary()
            
        #utils.plot_model(model, to_file='model.png')

        return model
    
    def load_weight_param(self, model, weight_path):
        model.load_weights(weight_path)
        model._make_predict_function()
        return model
