# %%
import sys, os
sys.path.append('../src')
sys.path.append(os.path.join(os.path.dirname(__file__)))

import math 

import torch
import torch.nn as nn
from torch import Tensor

import matplotlib.pyplot as plt



# %% 

def CNN_reg(feature_dim, output_dim=3):
    signal_input = Input(shape=feature_dim, name='Input')
    
    output = Conv1D(32, 3, strides=1, activation='relu')(signal_input)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    for i in range(3):
        output = Conv1D(32, 3, strides=1)(output)
        output = BatchNormalization(momentum=0.01)(output)
        output = Activation('relu')(output)
        output = MaxPool1D(2)(output)
        output = Dropout(0.25)(output)
        
    for i in range(0):
        output = Conv1D(32, 3, strides=1)(output)
        output = BatchNormalization(momentum=0.01)(output)
        output = Activation('relu')(output)
        #output = MaxPool1D(2)(output)
        output = Dropout(0.25)(output)
    
    output = Flatten()(output)                
    output = Dense(64, activation='relu')(output)
    output = Dense(output_dim, activation='linear')(output)

    model = Model(signal_input, outputs=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
        
    return model

def CNN_reg_big(feature_dim, output_dim=3):
    signal_input = Input(shape=feature_dim, name='Input')
    
    output = Conv1D(128, 3, strides=1)(signal_input)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
    
    output = Conv1D(128, 4, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)

    output = Conv1D(128, 5, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
    
    output = Conv1D(128, 4, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = Dropout(0.25)(output)
    
    output = Flatten()(output)                
    output = Dense(128, activation='relu')(output)
    output = Dense(output_dim, activation='linear')(output)

    model = Model(signal_input, outputs=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
        
    return model

def CNN_embed_reg(feature_dim, output_dim=3):
    signal_input = Input(shape=feature_dim, name='Input')
    
    output = Conv1D(128, 3, strides=1, activation='relu')(signal_input)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
    
    output = Conv1D(128, 3, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
    
    output = Conv1D(64, 3, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
        
    output = Conv1D(32, 3, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = Dropout(0.25)(output)
    
    output = layers.GlobalAveragePooling1D()(output)              
    output = Dense(64, activation='relu')(output)
    output = Dense(output_dim, activation='linear')(output)

    model = Model(signal_input, outputs=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
        
    return model

def lstm_pooling_reg(feature_dim=(200, 70), output_dim=3):
    model_input = layers.Input(shape=feature_dim, name='Input')
    #output = layers.GlobalAveragePooling1D()(model_input)
    
    output = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(model_input)

    output = layers.GlobalAveragePooling1D()(output)
    output = layers.Dense(32, activation='relu')(output)
    output = layers.Dense(output_dim, activation='linear')(output)

    model = Model(inputs=model_input, outputs=output)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse','mae'])

    return model
    
class ConvBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=False, maxpool=False,name='Convolutional block'):
        super(ConvBlock, self).__init__()
        
        self.id = 'convblock'
        
        if output_size==None:
            output_size=input_size
            
        self.conv = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)
        
        self.dropout = dropout
        self.maxpool = maxpool
        
        if dropout:
            self.dropout = nn.Dropout(0.25)
            
        if maxpool:
            self.nn_maxpool = nn.MaxPool1d(2)
        
    def forward(self, input):
        input = self.conv(input)
        input = self.activation(input)
        input = self.bn(input)
        
        if self.dropout:
            input = self.dropout(input)
        if self.maxpool:
            input = self.maxpool(input)
            
        return input
    
class CNNReg(nn.Module):
    def __init__(self, feature_dim, output_dim=3, name='Convolutional Regression'):
        super(CNNReg, self).__init__()
        
        self.id = 'CNNReg'

        self.conv = nn.Sequential(
            ConvBlock(feature_dim, 128),
            *[ConvBlock(128, 128) for _ in range(4)]
            )
        self.activation = nn.Sigmoid()
        
    def forward(self, input):
        input = self.conv(input)
        input = self.activation(input)
        return input
    
    

def CNN_reg_big(feature_dim, output_dim=3):
    signal_input = Input(shape=feature_dim, name='Input')
    
    output = Conv1D(128, 3, strides=1)(signal_input)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1, padding='same')(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    
    output = Conv1D(128, 3, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
    
    output = Conv1D(128, 4, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)

    output = Conv1D(128, 5, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = MaxPool1D(2)(output)
    output = Dropout(0.25)(output)
    
    output = Conv1D(128, 4, strides=1)(output)
    output = BatchNormalization(momentum=0.01)(output)
    output = Activation('relu')(output)
    output = Dropout(0.25)(output)
    
    output = Flatten()(output)                
    output = Dense(128, activation='relu')(output)
    output = Dense(output_dim, activation='linear')(output)

    model = Model(signal_input, outputs=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
        
    return model
