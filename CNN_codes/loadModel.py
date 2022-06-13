
import numpy as np

# from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/TestCircles/2D3D')

import numpy as np
import pandas as pd




def rotFlipComp2D():
    # cnn = load_model('2D_EKTEVARE_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation')#2D 5000 Trent på Torland-HC
    # cnn = load_model('2D_50000_CNNModelFor_LogLog_HCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 50000
    
    # cnn = load_model('2DCNNModelFor_LogLog_HCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 50000
    # cnn = load_model('2D_25000_CNNModelFor_LogLog_HCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 25000
    # cnn = load_model('2D_5000_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 5000
    cnn = load_model('2DCNNModelFor_LogLog_EKTE_VAREHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 50000 10000 Over/Under 30000 Middle

    # cnn = load_model('2D_Below50Voxels_ModelFor_LogLog_HCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 50000 16x16 Below50
    # cnn = load_model('2D_HCInterval_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN trained on images between -5 and -1
    
    # cnn = load_model('2D_Mellom-5-1_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D 5000 mellom intervall
    # cnn = load_model('2D_Over-1_Forsok2_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D 5000 over -1.17
    # cnn = load_model('2D_Under-5_Forsok2_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D 5000 under -4.87
    
 
    im = np.load('5000_Normalfordelt2DImages.npy', allow_pickle = True) #Opprinnelig bilde
    rot90 = np.load('2D90degmatrise5000.npy', allow_pickle = True)
    rot180 = np.load('2D180degmatrise5000.npy', allow_pickle = True)
    rot270 = np.load('2D270degmatrise5000.npy', allow_pickle = True)
    
    flip0 = np.load('2DFlipLRmatrise5000.npy', allow_pickle = True) #Flipped image, rotated 0 deg
    flip90 = np.load('2DFlipLR90degmatrise5000.npy', allow_pickle = True)
    flip180 = np.load('2DFlipLR180degmatrise5000.npy', allow_pickle = True)
    flip270 = np.load('2DFlipLR270degmatrise5000.npy', allow_pickle = True)

    hc = np.load('5000_NormalfordeltHC.npy', allow_pickle = True)
    print('Reshaping starter nå')
    XPix, YPix = int(np.sqrt(im.shape[1])), int(np.sqrt(im.shape[1]))
    im = im.reshape(-1, XPix, YPix, 1)
    rot90 = rot90.reshape(-1, XPix, YPix, 1)
    rot180 = rot180.reshape(-1, XPix, YPix, 1)
    rot270 = rot270.reshape(-1, XPix, YPix, 1)
    
    flip0 = flip0.reshape(-1, XPix, YPix, 1)
    flip90 = flip90.reshape(-1, XPix, YPix, 1)
    flip180 = flip180.reshape(-1, XPix, YPix, 1)
    flip270 = flip270.reshape(-1, XPix, YPix, 1)
    
    print('Prediksjoner starter nå')
    
    afPredictedim = cnn.predict(im)
    afPredictedrot90 = cnn.predict(rot90)
    afPredictedrot180 = cnn.predict(rot180)
    afPredictedrot270 = cnn.predict(rot270)
    afPredictedflip0 = cnn.predict(flip0)
    afPredictedflip90 = cnn.predict(flip90)
    afPredictedflip180 = cnn.predict(flip180)
    afPredictedflip270 = cnn.predict(flip270)
    
    print('Filene lagres')
    np.savetxt('2D_5000_CNN50000_PredictedLogHCIM.csv', afPredictedim)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCrot90.csv', afPredictedrot90)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCrot180.csv', afPredictedrot180)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCrot270.csv', afPredictedrot270)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCflip0.csv', afPredictedflip0)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCflip90.csv', afPredictedflip90)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCflip180.csv', afPredictedflip180)
    np.savetxt('2D_5000_CNN50000_PredictedLogHCflip270.csv', afPredictedflip270)

    
    print('Skriver til Excel')

    d = {'Labels': hc, 'Rot0deg': np.concatenate(afPredictedim), 'Rot90deg': np.concatenate(afPredictedrot90), \
         'Rot180deg': np.concatenate(afPredictedrot180), 'Rot270deg': np.concatenate(afPredictedrot270), 'Flip0deg': np.concatenate(afPredictedflip0), \
         'Flip90deg': np.concatenate(afPredictedflip90), 'Flip180deg': np.concatenate(afPredictedrot180), 'Flip270deg': np.concatenate(afPredictedrot270)}
    df = pd.DataFrame(data = d)

    filepath = 'Bok3.xlsx'

    df.to_excel(filepath, index = False)
    
    
    



def rotFlipComp3D():
    # cnn = load_model('3D_5000_CNNModelForLogHCEstimation300EpochsMAEWholeDataManualSplit3DWithoutRotation') #3D CNN 5000
    # cnn = load_model('3D_50000_CNNModelForLogHCEstimation300EpochsMAEWholeDataManualSplit3DWithoutRotation') #3D CNN 50000
    cnn = load_model('3D_25000_CNNModelForLogHCEstimation300EpochsMAEWholeDataManualSplit3DWithoutRotation') #3d CNN 25000
    
    im = np.load('5000_Normalfordelt3DImages.npy', allow_pickle = True) #Opprinnelig bilde
    rot90 = np.load('3D90degmatrise5000.npy', allow_pickle = True)
    rot180 = np.load('3D180degmatrise5000.npy', allow_pickle = True)
    rot270 = np.load('3D270degmatrise5000.npy', allow_pickle = True)
    
    flip0 = np.load('3DFlipLRmatrise5000.npy', allow_pickle = True) #Flipped image, rotated 0 deg
    flip90 = np.load('3DFlipLR90degmatrise5000.npy', allow_pickle = True)
    flip180 = np.load('3DFlipLR180degmatrise5000.npy', allow_pickle = True)
    flip270 = np.load('3DFlipLR270degmatrise5000.npy', allow_pickle = True)
    
    hc = np.load('5000_NormalfordeltHC.npy', allow_pickle = True)
    print('Reshaping starter nå')
    XPix, YPix,ZPix = im.shape[1], im.shape[2], im.shape[3]
    im = im.reshape(-1, XPix, YPix, ZPix, 1)
    rot90 = rot90.reshape(-1, XPix, YPix, ZPix, 1)
    rot180 = rot180.reshape(-1, XPix, YPix, ZPix, 1)
    rot270 = rot270.reshape(-1, XPix, YPix, ZPix, 1)
    
    flip0 = flip0.reshape(-1, XPix, YPix, ZPix, 1)
    flip90 = flip90.reshape(-1, XPix, YPix, ZPix, 1)
    flip180 = flip180.reshape(-1, XPix, YPix, ZPix, 1)
    flip270 = flip270.reshape(-1, XPix, YPix, ZPix, 1)
    
    print('Prediksjoner starter nå')
    
    afPredictedim = cnn.predict(im)
    afPredictedrot90 = cnn.predict(rot90)
    afPredictedrot180 = cnn.predict(rot180)
    afPredictedrot270 = cnn.predict(rot270)
    afPredictedflip0 = cnn.predict(flip0)
    afPredictedflip90 = cnn.predict(flip90)
    afPredictedflip180 = cnn.predict(flip180)
    afPredictedflip270 = cnn.predict(flip270)
    
    print('Filene lagres')
    np.savetxt('3D_25000_PredictedLogHCIM.csv', afPredictedim)
    np.savetxt('3D_25000_PredictedLogHCrot90.csv', afPredictedrot90)
    np.savetxt('3D_25000_PredictedLogHCrot180.csv', afPredictedrot180)
    np.savetxt('3D_25000_PredictedLogHCrot270.csv', afPredictedrot270)
    np.savetxt('3D_25000_PredictedLogHCflip0.csv', afPredictedflip0)
    np.savetxt('3D_25000_PredictedLogHCflip90.csv', afPredictedflip90)
    np.savetxt('3D_25000_PredictedLogHCflip180.csv', afPredictedflip180)
    np.savetxt('3D_25000_PredictedLogHCflip270.csv', afPredictedflip270)
    # np.savetxt('3D_50000_HC5000.csv', hc)
    print('Skriver til Excel')

    d = {'Labels': hc, 'Rot0deg': np.concatenate(afPredictedim), 'Rot90deg': np.concatenate(afPredictedrot90), \
         'Rot180deg': np.concatenate(afPredictedrot180), 'Rot270deg': np.concatenate(afPredictedrot270), 'Flip0deg': np.concatenate(afPredictedflip0), \
         'Flip90deg': np.concatenate(afPredictedflip90), 'Flip180deg': np.concatenate(afPredictedrot180), 'Flip270deg': np.concatenate(afPredictedrot270)}
    df = pd.DataFrame(data = d)

    filepath = 'Bok3.xlsx'

    df.to_excel(filepath, index = False)


def rotFlipComp32x32():
    
    # cnn = load_model('2D_50000_CNNModelFor_LogLog_HCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 50000
    # cnn = load_model('2D_25000_CNNModelFor_LogLog_HCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 25000
    # cnn = load_model('2D_5000_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 5000
    # cnn = load_model('2D_5000_32x32_CNNModelForLogHCEstimation300EpochsMAEMoreConUnrotatedValidation') #2D CNN 5000 32x32
    cnn = load_model('3D_32x32_CNNModelForLogHCEstimation300EpochsMAEWholeDataManualSplit3DWithoutRotation') #3D CNN 5000 32x32
   
    
    im = np.load('5000_32x32_Normalfordelt3DImages.npy', allow_pickle = True) #Opprinnelig bilde
    rot90 = np.load('3D_32x32_90degmatrise5000.npy', allow_pickle = True)
    rot180 = np.load('3D_32x32_180degmatrise5000.npy', allow_pickle = True)
    rot270 = np.load('3D_32x32_270degmatrise5000.npy', allow_pickle = True)
    
    flip0 = np.load('3D_32x32_FlipLRmatrise5000.npy', allow_pickle = True) #Flipped image, rotated 0 deg
    flip90 = np.load('3D_32x32_FlipLR90degmatrise5000.npy', allow_pickle = True)
    flip180 = np.load('3D_32x32_FlipLR180degmatrise5000.npy', allow_pickle = True)
    flip270 = np.load('3D_32x32_FlipLR270degmatrise5000.npy', allow_pickle = True)
 
    hc = np.load('5000_NormalfordeltHC.npy', allow_pickle = True)
    
 
    print('Reshaping starter nå')
    
    #####2D
    # XPix, YPix = int(np.sqrt(im.shape[1])), int(np.sqrt(im.shape[1]))
    # im = im.reshape(-1, XPix, YPix, 1)
    # rot90 = rot90.reshape(-1, XPix, YPix, 1)
    # rot180 = rot180.reshape(-1, XPix, YPix, 1)
    # rot270 = rot270.reshape(-1, XPix, YPix, 1)
    
    # flip0 = flip0.reshape(-1, XPix, YPix, 1)
    # flip90 = flip90.reshape(-1, XPix, YPix, 1)
    # flip180 = flip180.reshape(-1, XPix, YPix, 1)
    # flip270 = flip270.reshape(-1, XPix, YPix, 1)
    
    ######3D
    XPix, YPix, Zpix = im.shape[1], im.shape[2], im.shape[3]
    im = im.reshape(-1, XPix, YPix, Zpix, 1)
    rot90 = rot90.reshape(-1, XPix, YPix, Zpix, 1)
    rot180 = rot180.reshape(-1, XPix, YPix, Zpix, 1)
    rot270 = rot270.reshape(-1, XPix, YPix, Zpix, 1)
    
    flip0 = flip0.reshape(-1, XPix, YPix, Zpix, 1)
    flip90 = flip90.reshape(-1, XPix, YPix, Zpix, 1)
    flip180 = flip180.reshape(-1, XPix, YPix, Zpix, 1)
    flip270 = flip270.reshape(-1, XPix, YPix, Zpix, 1)
    
    print('Prediksjoner starter nå')
    
    afPredictedim = cnn.predict(im)
    afPredictedrot90 = cnn.predict(rot90)
    afPredictedrot180 = cnn.predict(rot180)
    afPredictedrot270 = cnn.predict(rot270)
    afPredictedflip0 = cnn.predict(flip0)
    afPredictedflip90 = cnn.predict(flip90)
    afPredictedflip180 = cnn.predict(flip180)
    afPredictedflip270 = cnn.predict(flip270)
    
    print('Filene lagres')
    # np.savetxt('2D_5000_32x32_PredictedLogHCIM.csv', afPredictedim)
    # np.savetxt('2D_5000_32x32_PredictedLogHCrot90.csv', afPredictedrot90)
    # np.savetxt('2D_5000_32x32_PredictedLogHCrot180.csv', afPredictedrot180)
    # np.savetxt('2D_5000_32x32_PredictedLogHCrot270.csv', afPredictedrot270)
    # np.savetxt('2D_5000_32x32_PredictedLogHCflip0.csv', afPredictedflip0)
    # np.savetxt('2D_5000_32x32_PredictedLogHCflip90.csv', afPredictedflip90)
    # np.savetxt('2D_5000_32x32_PredictedLogHCflip180.csv', afPredictedflip180)
    # np.savetxt('2D_5000_32x32_PredictedLogHCflip270.csv', afPredictedflip270)
    # np.savetxt('2D_5000_HC2000.csv', hc)
    print('Skriver til Excel')

    d = {'Labels': hc, 'Rot0deg': np.concatenate(afPredictedim), 'Rot90deg': np.concatenate(afPredictedrot90), \
         'Rot180deg': np.concatenate(afPredictedrot180), 'Rot270deg': np.concatenate(afPredictedrot270), 'Flip0deg': np.concatenate(afPredictedflip0), \
         'Flip90deg': np.concatenate(afPredictedflip90), 'Flip180deg': np.concatenate(afPredictedrot180), 'Flip270deg': np.concatenate(afPredictedrot270)}
    df = pd.DataFrame(data = d)

    filepath = 'Bok3.xlsx'

    df.to_excel(filepath, index = False)
# rotFlipComp32x32_2D()


def fullSetPred():
    
    cnn = load_model('2D_50000_CNNModelFor_LogLog_HCEstimation300Epochs') #2D CNN 50000

    im = np.load('2DFullSetImages.npy', allow_pickle = True)
    # hc = np.load('5000HC.npy', allow_pickle = True)
    print('Reshaping starter nå')
    XPix, YPix = int(np.sqrt(im.shape[1])), int(np.sqrt(im.shape[1]))
    im = im.reshape(-1, XPix, YPix, 1)
    print('Prediksjoner starter nå')
    afPredictedim = cnn.predict(im)
    print('Filene lagres')
    np.save('Predicted_hc_FULLSET.npy', afPredictedim)





