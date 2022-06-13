# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:35:23 2022

@author: lavra
"""
from properties import nCrossCube, n_pCube, paddSize, topnbottom, side
import numpy as np


def padder2D(image, hc):
  
    start = 0
    step = n_pCube

    paddedIm = []
    prIm = int(n_pCube*np.sqrt(nCrossCube))
    
    paddedIm.extend(topnbottom*int(np.sqrt(nCrossCube))) #TopnBottom er en liste med 1'ere, like lang som n_pCube + padding på hver side
    total = int(n_pCube*nCrossCube) #iterasjonen tar med hvert fjerde tall
    it = int(np.sqrt(nCrossCube))
    
    for i in range(0,total,it): #fra-til-steg
        
        if i in {prIm, 2*prIm, 3*prIm, 4*prIm}:
            paddedIm.extend(topnbottom*int(np.sqrt(nCrossCube))) 
            paddedIm.extend(topnbottom*int(np.sqrt(nCrossCube))) 
            
        paddedIm.extend(side)
        paddedIm.extend(image[start+step*i:step*(i+1)]) #0-32
        paddedIm.extend(side)
        paddedIm.extend(side)
        paddedIm.extend(image[start+step*i+n_pCube:step*(i+1)+n_pCube]) #32-64
        paddedIm.extend(side)
        paddedIm.extend(side)
        paddedIm.extend(image[start+step*i+2*n_pCube:step*(i+1)+2*n_pCube]) #64-96
        paddedIm.extend(side)
        paddedIm.extend(side)
        paddedIm.extend(image[start+step*i+3*n_pCube:step*(i+1)+3*n_pCube]) #96 -128
        paddedIm.extend(side)
        # print(len(paddedIm))
       
    paddedIm.extend(topnbottom*int(np.sqrt(nCrossCube))) 
  
    hc = hc*(n_pCube)**2/(2*paddSize + n_pCube)**2
    
    return paddedIm, hc



def padder3D(image): #Tar inn en array av bilder. 
    imStr = []
    start = 0
    step = n_pCube
    for im in image:
        paddedIm = []
        paddedIm.extend(topnbottom)

        for i in range(0, n_pCube):
            
            paddedIm.extend(side)
            paddedIm.extend(im[start + i*step : step*(i+1)])
            paddedIm.extend(side)
            
        paddedIm.extend(topnbottom)
        imStr.append(paddedIm)
    return imStr
    