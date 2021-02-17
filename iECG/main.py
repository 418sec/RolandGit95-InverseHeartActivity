# execute with qsub main.py -config config/config_cnn.yaml 

#! python

# name
#$ -N iECG_CNN

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/torch/bin/python

# Merge error and out
#$ -j yes

# Path for output
#$ -o /home/stenger/smaxxhome/outputs

import os, sys
from tqdm import tqdm
import yaml

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import tensorflow as tf

import glob
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


modelFolder = '../../src/modeling/'
sys.path.append(modelFolder)

import wandb
from wandb.keras import WandbCallback

from omegaconf import DictConfig, OmegaConf
import argparse

wandb.login()

# %%
def CNN_reg(feature_dim, config, output_dim=3):
    signal_input = layers.Input(shape=feature_dim, name='Input')
    output = signal_input

    # Feature extraction
    for i in range(3):
        output = layers.Conv1D(config.cnn_features, 3, activation='relu')(output)
        output = layers.BatchNormalization(momentum=0.01)(output)
        output = layers.MaxPool1D(2)(output)
        output = layers.Dropout(0.25)(output)
        
    for _ in range(config.cnn_layers):
	    output = layers.Conv1D(config.cnn_features, 3, activation='relu', padding='same')(output)
	    output = layers.BatchNormalization(momentum=0.01)(output)

    # Regression
    output = layers.Flatten()(output)                
    for _ in range(config.reg_layers):
    	output = layers.Dense(config.reg_features, activation='relu')(output)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.Dense(output_dim, activation='linear')(output)
    
    if config.optimizer=='Adam':
        optimizer = optimizers.Adam(config.lr)
    elif config.optimizer=='SGD':
        optimizer = optimizers.SGD(config.lr)
        
    if config.loss_fn=='MSE':
        loss = losses.mean_squared_error
    elif config.loss_fn=='MAE':
        loss = losses.mean_absolute_error      
        
    model = models.Model(signal_input, outputs=output)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
            
    return model

# %%

def pipeline(config):
    model = CNN_reg((100,70), config)
    model.summary()

    es = callbacks.EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=32)
    rp = callbacks.ReduceLROnPlateau(monitor='val_mae', patience=8)
    mc = callbacks.ModelCheckpoint("models/weights/RegCNN", monitor='val_mae', mode='min', save_best_only=True)

    
    folder = 'data/processed/'
    files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    
    X_train = np.load(folder + files[0])
    y_train = np.load(folder + files[1])
    
    X_test = np.load(folder + files[2])
    y_test = np.load(folder + files[3])

    if config.wandb:
        wandb.init(project="CNNREG")
        cbks=[es, mc, rp, WandbCallback()]
    else:
        cbks=[es, mc, rp]
    
    model.fit(X_train[:,:100], y_train, 
                epochs=config.epochs, 
                batch_size=config.batch_size, 
                verbose=1, 
                callbacks=cbks,
                validation_data=(X_test[:,:100], y_test))
    
def pipeline_sweep(config):
    model = CNN_reg((100,70), config)
    model.summary()

    es = callbacks.EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=32)
    rp = callbacks.ReduceLROnPlateau(monitor='val_mae', patience=8)
    mc = callbacks.ModelCheckpoint("models/weights/RegCNN", monitor='val_mae', mode='min', save_best_only=True)

    
    folder = 'data/processed/'
    files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    
    X_train = np.load(folder + files[0])
    y_train = np.load(folder + files[1])
    
    X_test = np.load(folder + files[2])
    y_test = np.load(folder + files[3])

    if config.wandb:
        wandb.init(project="CNNREG")
        cbks=[es, mc, rp, WandbCallback()]
    else:
        cbks=[es, mc, rp]
    
    model.fit(X_train[:,:100], y_train, 
                epochs=config.epochs, 
                batch_size=config.batch_size, 
                verbose=1, 
                callbacks=cbks,
                validation_data=(X_test[:,:100], y_test))

# %%

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')

    # Names
    parser.add_argument('-project_name', '--project_name', type=str, help='', default='unnamed')
    parser.add_argument('-name', '--name', type=str, help='', default='unnamed')

    # Training process
    parser.add_argument('-epochs', '--epochs', type=int, help='', default=1)
    parser.add_argument('-batch_size', '--batch_size', type=int, help='', default=3)
    parser.add_argument('-lr', '--lr', type=float, help='Learning Rate', default=0.001)
    
    parser.add_argument('-loss_fn', '--loss_fn', type=str, help='', default='MSE')
    parser.add_argument('-optimizer', '--optimizer', type=str, help='', default='Adam')
    
    # Config files
    parser.add_argument('-config', '--config', type=str, help='Place of config file', default=None)
    
    # Logger
    parser.add_argument('-wandb', '--wandb', type=int, help='', default=True)
    parser.add_argument('-online', '--online', type=int, help='', default=True)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, help='', default='./')
    
    # Load and save
    parser.add_argument('-weights_file', '--weights_file', type=str, help='If load_weights==True, set the location of the weights file', default='./model')
    parser.add_argument('-save_name', '--save_name', type=str, help='', default='model')
    parser.add_argument('-save_dir', '--save_dir', type=str, help='', default='./')
    
    parser.add_argument('-dataset_dir', '--dataset_dir', type=str, help='', default=None)
    parser.add_argument('-sweep', '--sweep', type=bool, default=False)
    
    parser.add_argument('-cnn_features', '--cnn_features', type=int, default=256)
    parser.add_argument('-reg_features', '--reg_features', type=int, default=256)
    parser.add_argument('-cnn_layers', '--cnn_layers', type=int, default=1)
    parser.add_argument('-reg_layers', '--reg_layers', type=int, default=1)

    
    args = parser.parse_args()    
    args_config = vars(args)
    
    if int(args_config['online'])==0:
        print('No internet')
        os.environ['WANDB_MODE'] = 'dryrun'
        WANDB_MODE="dryrun"

    specified_config = dict()
    for key, value in vars(args).items():
        if parser.get_default(key)!=value:
            specified_config[key] = value    
    
    if not isinstance(args.config, type(None)):
        try:
            with open(args.config) as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                args_config.update(config)
        except FileNotFoundError:
            print('Config-file not found, use default values')
            assert('Config-file not found, use default values')     
            
    args_config.update(specified_config)
    
    args_config = OmegaConf.create(args_config)
    
    if args_config.sweep:
        print(args_config)
        pipeline_sweep(args_config)
    else:
        #args_config.cnn_features = 256
        #a#rgs_config.reg_features = 256
        #args_config.cnn_layers = 1
        #args_config.reg_layers = 2
        print(args_config)

        pipeline(args_config)
    #for key, value in args_config.items():
        #print(key + ':', value)
    
    


