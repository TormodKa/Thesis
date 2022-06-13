#!/usr/bin/env python
# coding: utf-8

#This is a modified version of a code for building a 2D CNN model used for porosity estimation. 
#The original code was made by Kurdistan Chawshin. 


import numpy as np 
seed = 7
np.random.seed(seed)
import os 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from tensorflow import keras
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import time
t0 = time.time()



im = np.load('codeOutputs/TrainTestValid/2D3D/NormDist5000/5000TTV_3D.npy', allow_pickle = True)
hc = np.load('codeOutputs/TrainTestValid/2D3D/NormDist5000/5000HC_3D.npy', allow_pickle = True)
aaaiTrainImages, afTrainHC = im[0:5000], hc[0:5000]
aaaiValidationImages, afValidationHC = im[5000:6000], hc[5000:6000]
aaaiTestImages, afTestHC = im[6000:7000], hc[6000:7000]



batch_size = 32
epochs = 10




XPix, YPix,ZPix = aaaiTrainImages.shape[1], aaaiTrainImages.shape[2],aaaiTrainImages.shape[3]
aaaiTrainImages = aaaiTrainImages.reshape(-1,XPix,YPix,ZPix,1)
aaaiValidationImages = aaaiValidationImages.reshape(-1,XPix,YPix,ZPix,1)
aaaiTestImages = aaaiTestImages.reshape(-1,XPix,YPix,ZPix,1)




from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from sklearn.metrics import mean_squared_error 
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ReLU, GlobalAveragePooling3D
from tensorflow.keras import activations
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters




def ModelCreation(hp):
    Regressor = keras.Sequential()
    
    Regressor.add(keras.layers.Conv3D(
            filters= hp.Int('ConvFiltersInputLayer', min_value = 16, max_value=256,step=32),
            kernel_size=hp.Choice('Conv2Kernel', values=[3,5]), padding='same',input_shape =(XPix, YPix,ZPix,1)))

    Regressor.add(keras.layers.ReLU())
    Regressor.add(keras.layers.MaxPooling3D(
            pool_size=(2,2,2),padding='same'))
    
    for i in range(hp.Int('ConvBlocks', 1, 3, default=1)): 
        Regressor.add(keras.layers.Conv3D(
            filters= hp.Int(f'Conv{i}_Filter', min_value = 16, max_value=256, step=32),
            kernel_size= hp.Choice('Conv2Kernel', values=[3,5]), padding='same'))
    
        Regressor.add(keras.layers.ReLU())
        Regressor.add(keras.layers.MaxPooling3D(pool_size=(2,2,2),padding='same'))

    
    Regressor.add(keras.layers.Flatten())
        
    Regressor.add(keras.layers.Dense(
            units= hp.Int('HiddenNeurons', 32, 256, step=32),
            activation='linear'))
    
    Regressor.add(keras.layers.ReLU()) 
    Regressor.add(keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.6,default=0.00,step=0.2)))
    
    
    
    Regressor.add(keras.layers.Dense(1, activation='linear'))
    

    Regressor.compile(loss="mean_absolute_error", optimizer=keras.optimizers.Adam(
        hp.Choice('LearningRate',[1e-2, 1e-3, 1e-4])))
    return Regressor

            
    



EarlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=5)



Tuner = Hyperband(ModelCreation, objective= 'val_loss', max_epochs =epochs, executions_per_trial =1, seed=seed, directory=os.path.normpath('codeOutputs/TrainTestValid/2D3D/codeOutputs5000/3DTransportPropLog'), project_name='3DHyperparameter')

Tuner.search(aaaiTrainImages, afTrainHC, verbose=2, validation_data= (aaaiValidationImages, afValidationHC), epochs=epochs, callbacks=[EarlyStop])

Tuner.search_space_summary()

Model = Tuner.get_best_models(1)[0]

print(Model.summary())

Tuner.results_summary()

print(Tuner.get_best_hyperparameters(1)[0].values)



Epochs = 300
EarlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=30)


BestRegressor = Model.fit(aaaiTrainImages, afTrainHC, validation_data = (aaaiValidationImages, afValidationHC), epochs=Epochs, callbacks =[EarlyStop], verbose=2)


t1 = time.time()
print('Training took: ',(t1 - t0)/60,'minutes')


TestLoss  = Model.evaluate(aaaiTestImages, afTestHC, verbose=2)
print('Test loss:', TestLoss)


plt.plot(BestRegressor.history['loss'], color ='b')
plt.plot(BestRegressor.history['val_loss'], color = 'r')
#plt.title('model loss')
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.savefig('codeOutputs/TrainTestValid/2D3D/codeOutputs5000/3DGridImagesLog_16x16_5000_1000_1000.png',  dpi = 300)




afPredictedHC = Model.predict(aaaiTestImages)
np.savetxt('codeOutputs/TrainTestValid/2D3D/codeOutputs5000/3D_PredictedLogHC_GridImages_5000_1000_1000.csv', afPredictedHC)
save_model(Model, filepath= 'codeOutputs/TrainTestValid/2D3D/codeOutputs5000/Models/HCEstimation/3D_CNNModelForLogHCEstimation300Epochs', include_optimizer=True)



