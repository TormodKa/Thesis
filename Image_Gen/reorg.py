# -*- coding: utf-8 -*-


import pointToGridValue
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import properties
from padding import padder2D, padder3D



def reorg(iNodeA, iNodeB, cubePoints, nCrossSections):
    nSides = int(sqrt(len(cubePoints)/(nCrossSections)))
    reorgImg = []
    lst = [0]*int(nCrossSections**(0.5))
    nLines = nSides*int(nCrossSections**(0.5))
    pInSquare = nSides**2
    for num in range(1, len(lst)):
        lst[num] = lst[num-1] + pInSquare

    cont = True
    start = [0]*nLines
    end = [nSides]*nLines
    for i in range(1, len(start)):
        start[i] = start[i-1]+nSides
        end[i] = end[i-1]+nSides
        if i % nSides == 0:
            start[i] = start[i]+int(nCrossSections**(0.5)-1)*nSides**2
            end[i] = end[i] + int(nCrossSections**(0.5)-1)*nSides**2

    st = 0
    ed = 0
    i = 0
 
    while cont == True:

        for i in range(0, len(lst)):
            reorgImg.extend(cubePoints[(start[st]+lst[i]):(end[ed]+lst[i])])


        st += 1
        ed += 1

        if st == (nLines):
            cont = False
            
  return reorgImg


def reorg3D(iNodeA, iNodeB, cubePoints, nCrossSections):
    reorg3DImg = []

    pointsInImg = int(len(cubePoints)/nCrossSections)
    for i in range(0, nCrossSections):
        p = cubePoints[0+i*pointsInImg:pointsInImg+i*pointsInImg]
        a = np.array(p).reshape(properties.n_pCube, properties.n_pCube)
        reorg3DImg.append(a)

    return reorg3DImg


def rotator(train3D, valid3D, test3D):
    rotIm3D90 = np.rot90(train3D, axes=(-2, -1))
    rotIm3D180 = np.rot90(rotIm3D90, axes=(-2, -1))
    rotIm3D270 = np.rot90(rotIm3D180, axes=(-2, -1))
    
    flip3DIm = []
    
    for i in range(len(train3D)):
        flip3DIm.append(np.fliplr(train3D[i]))
        
    
    flip3DIm90 = np.rot90( flip3DIm, axes = (-2,-1))
    flip3DIm180 = np.rot90( flip3DIm90, axes = (-2,-1))
    flip3DIm270 = np.rot90(flipIm3D180, axes=(-2, -1))

    cP = properties.nCrossCube

   
    rot2D90, rot2D180, rot2D270 = [],[],[]
    flip2D90, flip2D180, flip2D270 = [],[],[]
    for i in range(len(train3D)):
        rot2D90.append(np.concatenate(np.concatenate(rotIm3D90[i])))
        rot2D180.append(np.concatenate(np.concatenate(rotIm3D180[i])))
        rot2D270.append(np.concatenate(np.concatenate(rotIm3D270[i])))        
        
        flip2D90.append(np.concatenate(np.concatenate(flipIm3D90[i])))
        flip2D180.append(np.concatenate(np.concatenate(flipIm3D180[i])))
        flip2D270.append(np.concatenate(np.concatenate(flipIm3D270[i]))) 
    
    rotIm2D90, rotIm2D180, rotIm2D270 = [],[],[]
    flipIm2D90, flipIm2D180, flipIm2D270 = [],[],[]
    
    for i in range(len(train3D)):
        rotIm2D90.append(reorg('x', 'x', rot2D90[i], cP))
        rotIm2D180.append(reorg('x', 'x', rot2D180[i], cP))
        rotIm2D270.append(reorg('x', 'x', rot2D270[i], cP))
    
    
    
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D90degmatriseTrening'+str(len(train3D))+'.npy', np.array(rotIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D180degmatriseTrening'+str(len(train3D))+'.npy', np.array(rotIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D270degmatriseTrening'+str(len(train3D))+'.npy', np.array(rotIm2D270))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D90degmatriseTrening'+str(len(train3D))+'.npy', np.array(rotIm3D90))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D180degmatriseTrening'+str(len(train3D))+'.npy', np.array(rotIm3D180))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D270degmatriseTrening'+str(len(train3D))+'.npy', np.array(rotIm3D270))
    
    
    
    rotIm3D90 = np.rot90(valid3D, axes=(-2, -1))
    rotIm3D180 = np.rot90(rotIm3D90, axes=(-2, -1))
    rotIm3D270 = np.rot90(rotIm3D180, axes=(-2, -1))

    cP = properties.nCrossCube

   
    rot2D90, rot2D180, rot2D270 = [],[],[]
    for i in range(len(valid3D)):
        rot2D90.append(np.concatenate(np.concatenate(rotIm3D90[i])))
        rot2D180.append(np.concatenate(np.concatenate(rotIm3D180[i])))
        rot2D270.append(np.concatenate(np.concatenate(rotIm3D270[i])))        
        
    
    rotIm2D90, rotIm2D180, rotIm2D270 = [],[],[]
    
    for i in range(len(valid3D)):
        rotIm2D90.append(reorg('x', 'x', rot2D90[i], cP))
        rotIm2D180.append(reorg('x', 'x', rot2D180[i], cP))
        rotIm2D270.append(reorg('x', 'x', rot2D270[i], cP))
    
    
    
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D90degmatriseValid'+str(len(valid3D))+'.npy', np.array(rotIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D180degmatriseValid'+str(len(valid3D))+'.npy', np.array(rotIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D270degmatriseValid'+str(len(valid3D))+'.npy', np.array(rotIm2D270))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D90degmatriseValid'+str(len(valid3D))+'.npy', np.array(rotIm3D90))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D180degmatriseValid'+str(len(valid3D))+'.npy', np.array(rotIm3D180))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D270degmatriseValid'+str(len(valid3D))+'.npy', np.array(rotIm3D270))
    
    
    rotIm3D90 = np.rot90(test3D, axes=(-2, -1))
    rotIm3D180 = np.rot90(rotIm3D90, axes=(-2, -1))
    rotIm3D270 = np.rot90(rotIm3D180, axes=(-2, -1))

    cP = properties.nCrossCube
 
   
    rot2D90, rot2D180, rot2D270 = [],[],[]
    for i in range(len(test3D)):
        rot2D90.append(np.concatenate(np.concatenate(rotIm3D90[i])))
        rot2D180.append(np.concatenate(np.concatenate(rotIm3D180[i])))
        rot2D270.append(np.concatenate(np.concatenate(rotIm3D270[i])))        
        
    
    rotIm2D90, rotIm2D180, rotIm2D270 = [],[],[]
    
    for i in range(len(test3D)):
        rotIm2D90.append(reorg('x', 'x', rot2D90[i], cP))
        rotIm2D180.append(reorg('x', 'x', rot2D180[i], cP))
        rotIm2D270.append(reorg('x', 'x', rot2D270[i], cP))
    
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D90degmatriseTest'+str(len(test3D))+'.npy', np.array(rotIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D180degmatriseTest'+str(len(test3D))+'.npy', np.array(rotIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/2D/2D270degmatriseTest'+str(len(test3D))+'.npy', np.array(rotIm2D270))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D90degmatriseTest'+str(len(test3D))+'.npy', np.array(rotIm3D90))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D180degmatriseTest'+str(len(test3D))+'.npy', np.array(rotIm3D180))
    np.save('codeOutputs/TrainTestValid/2D3D/'+str(len(train3D))+'/Rotations/3D/3D270degmatriseTest'+str(len(test3D))+'.npy', np.array(rotIm3D270))
    



def rotFullSet():
    
    # im = np.load('codeOutputs/TrainTestValid/2D3D/FullSet/3DFullSetImages.npy', allow_pickle = True)
    im = np.load('codeOutputs/TrainTestValid/2D3D/5000_32x32_Normalfordelt3DImages.npy', allow_pickle = True)
    rotIm3D90 = np.rot90(im, axes=(-2, -1))
    rotIm3D180 = np.rot90(rotIm3D90, axes=(-2, -1))
    rotIm3D270 = np.rot90(rotIm3D180, axes=(-2, -1))
    
    flip3DIm = []
    
    for i in range(len(im)):
        flip3DIm.append(np.fliplr(im[i]))
        
    
    flip3DIm90 = np.rot90( flip3DIm, axes = (-2,-1))
    flip3DIm180 = np.rot90( flip3DIm90, axes = (-2,-1))
    flip3DIm270 = np.rot90(flip3DIm180, axes=(-2, -1))
    
    
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_90degmatrise5000.npy', np.array(rotIm3D90))
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_180degmatrise5000.npy', np.array(rotIm3D180))
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_270degmatrise5000.npy', np.array(rotIm3D270))
    
    
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_FlipLRmatrise5000.npy', np.array(flip3DIm))
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_FlipLR90degmatrise5000.npy', np.array(flip3DIm90))
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_FlipLR180degmatrise5000.npy', np.array(flip3DIm180))
    np.save('codeOutputs/TrainTestValid/2D3D/3D_32x32_FlipLR270degmatrise5000.npy', np.array(flip3DIm270))


    rotIm = im
    cP = properties.nCrossCube
    rotIm = np.concatenate(np.concatenate(np.concatenate(rotIm)))

    im2D = reorg('x','x', rotIm, cP)
   
    rot2D90, rot2D180, rot2D270 = [],[],[]
    flip2D, flip2D90, flip2D180, flip2D270 = [],[],[],[]
    for i in range(len(im)):
        rot2D90.append(np.concatenate(np.concatenate(rotIm3D90[i])))
        rot2D180.append(np.concatenate(np.concatenate(rotIm3D180[i])))
        rot2D270.append(np.concatenate(np.concatenate(rotIm3D270[i])))        
        
        flip2D.append(np.concatenate(np.concatenate(flip3DIm[i])))
        flip2D90.append(np.concatenate(np.concatenate(flip3DIm90[i])))
        flip2D180.append(np.concatenate(np.concatenate(flip3DIm180[i])))
        flip2D270.append(np.concatenate(np.concatenate(flip3DIm270[i]))) 
    
    rotIm2D90, rotIm2D180, rotIm2D270 = [],[],[]
    flipIm2D, flipIm2D90, flipIm2D180, flipIm2D270 = [],[],[], []
    
    for i in range(len(im)):
        rotIm2D90.append(reorg('x', 'x', rot2D90[i], cP))
        rotIm2D180.append(reorg('x', 'x', rot2D180[i], cP))
        rotIm2D270.append(reorg('x', 'x', rot2D270[i], cP))    
        
        flipIm2D.append(reorg('x', 'x', flip2D[i], cP))
        flipIm2D90.append(reorg('x', 'x', flip2D90[i], cP))
        flipIm2D180.append(reorg('x', 'x', flip2D180[i], cP))
        flipIm2D270.append(reorg('x', 'x', flip2D270[i], cP))    
    
    
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_90degmatrise5000.npy', np.array(rotIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_180degmatrise5000.npy', np.array(rotIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_270degmatrise5000.npy', np.array(rotIm2D270))
   
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_FlipLRmatrise5000.npy', np.array(flipIm2D))
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_FlipLR90degmatrise5000.npy', np.array(flipIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_FlipLR180degmatrise5000.npy', np.array(flipIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/2D_32x32_FlipLR270degmatrise5000.npy', np.array(flipIm2D270))
   
    



def flipperspill():
    f3d0 = np.load('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/3D/3DFlipLRmatriseFullSet.npy', allow_pickle = True)
    f3d90 = np.load('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/3D/3DFlipLR90degmatriseFullSet.npy', allow_pickle = True)
    f3d180 = np.load('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/3D/3DFlipLR180degmatriseFullSet.npy', allow_pickle = True)
    f3d270 = np.load('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/3D/3DFlipLR270degmatriseFullSet.npy', allow_pickle = True)
    if len(f3d0) =={len(f3d90),len(f3d180),len(f3d270)}:
        l = len(f3d0)
        print(l)
    cP = properties.nCrossCube
    flip2D, flip2D90, flip2D180, flip2D270 = [],[],[],[]
    for i in range(l):
       
        flip2D.append(np.concatenate(np.concatenate(f3d0[i])))
        flip2D90.append(np.concatenate(np.concatenate(f3d90[i])))
        flip2D180.append(np.concatenate(np.concatenate(f3d180[i])))
        flip2D270.append(np.concatenate(np.concatenate(f3d270[i]))) 
    
    flipIm2D, flipIm2D90, flipIm2D180, flipIm2D270 = [],[],[], []
    
    for i in range(l):
        flipIm2D.append(reorg('x', 'x', flip2D[i], cP))
        flipIm2D90.append(reorg('x', 'x', flip2D90[i], cP))
        flipIm2D180.append(reorg('x', 'x', flip2D180[i], cP))
        flipIm2D270.append(reorg('x', 'x', flip2D270[i], cP))


    np.save('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/2D/2DFlipLRmatriseFullSet.npy', np.array(flipIm2D))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/2D/2DFlipLR90degmatriseFullSet.npy', np.array(flipIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/2D/2DFlipLR180degmatriseFullSet.npy', np.array(flipIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet/Rotations/2D/2DFlipLR270degmatriseFullSet.npy', np.array(flipIm2D270))




def rot_flip_32x32():
    im = np.load('codeOutputs/TrainTestValid/2D3D/FullSet32x32/3DFullSetImages.npy')
    im2D = np.load('codeOutputs/TrainTestValid/2D3D/FullSet32x32/2DFullSetImages.npy')
    im2D = im2D[100000:150000]
    im = im[100000:150000]
    l = len(im)
    rotIm3D90 = np.rot90(im, axes=(-2, -1))
    rotIm3D180 = np.rot90(rotIm3D90, axes=(-2, -1))
    rotIm3D270 = np.rot90(rotIm3D180, axes=(-2, -1))
    
    flip3DIm = []
    for i in range(len(im)):
        flip3DIm.append(np.fliplr(im[i]))
        
    
    flip3DIm90 = np.rot90( flip3DIm, axes = (-2,-1))
    flip3DIm180 = np.rot90( flip3DIm90, axes = (-2,-1))
    flip3DIm270 = np.rot90(flip3DIm180, axes=(-2, -1))
    
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DRot0degImages.npy', im)
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DRot90degImages.npy', rotIm3D90)
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DRot180degImages.npy', rotIm3D180)
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DRot270degImages.npy', rotIm3D270)
    
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DFlipLR0degImages.npy', flip3DIm)
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DFlipLR90degImages.npy', flip3DIm90)
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DFlipLR180degImages.npy', flip3DIm180)
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/3DFlipLR270degImages.npy', flip3DIm270)
    im3D = rotIm
    
    cP = properties.nCrossCube
    
    rot2D90, rot2D180, rot2D270 = [], [], []
    flip2D, flip2D90, flip2D180, flip2D270 = [],[],[],[]
    for i in range(l):
        rot2D90.append(np.concatenate(np.concatenate(rotIm3D90[i])))
        rot2D180.append(np.concatenate(np.concatenate(rotIm3D180[i])))
        rot2D270.append(np.concatenate(np.concatenate(rotIm3D270[i]))) 
       
        flip2D.append(np.concatenate(np.concatenate(flip3DIm[i])))
        flip2D90.append(np.concatenate(np.concatenate(flip3DIm90[i])))
        flip2D180.append(np.concatenate(np.concatenate(flip3DIm180[i])))
        flip2D270.append(np.concatenate(np.concatenate(flip3DIm270[i]))) 
    
    flipIm2D, flipIm2D90, flipIm2D180, flipIm2D270 = [],[],[],[]
    rotIm2D90, rotIm2D180, rotIm2D270 = [],[],[]
    for i in range(l):
        
        rotIm2D90.append(reorg('x', 'x', rot2D90[i], cP))
        rotIm2D180.append(reorg('x', 'x', rot2D180[i], cP))
        rotIm2D270.append(reorg('x', 'x', rot2D270[i], cP)) 
        
        flipIm2D.append(reorg('x', 'x', flip2D[i], cP))
        flipIm2D90.append(reorg('x', 'x', flip2D90[i], cP))
        flipIm2D180.append(reorg('x', 'x', flip2D180[i], cP))
        flipIm2D270.append(reorg('x', 'x', flip2D270[i], cP))
    
    
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DRot0degImages.npy', np.array(im2D))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DRot90degImages.npy', np.array(rotIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DRot180degImages.npy', np.array(rotIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DRot270degImages.npy', np.array(rotIm2D270))
   
    
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DFlipLR0degImages.npy', np.array(flipIm2D))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DFlipLR90degImages.npy', np.array(flipIm2D90))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DFlipLR180degImages.npy', np.array(flipIm2D180))
    np.save('codeOutputs/TrainTestValid/2D3D/FullSet32x32/Rotations/2DFlipLR270degImages.npy', np.array(flipIm2D270))





