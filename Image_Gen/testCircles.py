

import numpy as np
import math as m
import random
import matplotlib.pyplot as plt  
from skimage import draw
from properties import n_pCube, paddSize, nCrossCube
from reorg import reorg3D, reorg
from padding import padder2D

# #---------------- Framgangsmåte -----------------
# #Tilfeldige radiuser og lengder
# #Arealet av bildet er den totale lengden i andre. 
# #Arealet av hver tube pr totalt areal avgjør hvor stort bildet av tuben blir
# #Hagen-Poiseuille for å regne ut g. Skaleres ved Atot^2 og Ltot.
# #Finner forhold mellom lengdene, for å bestemme antall bilder fra hvert areal


def g(rinsc):
    visc = 1
    G = 1/m.pi/4
    return 3/80*rinsc**4/visc/G



def gtot(rinsc1, l1, rinsc2, l2, rinsc3, l3): 

    
    g1, g2, g3 = g(rinsc1), g(rinsc2),  g(rinsc3)

    Ltot = l1 + l2 + l3
    g123 = Ltot*pow(l1/g1 + l2/g2 + l3/g3, -1)
    Atot = Ltot**2
    g123 = g123/(Atot**2) 
    return g123


def circleHC(rinsc1, l1, rinsc2, l2, rinsc3, l3, ver):
    g123 = gtot(rinsc1, l1, rinsc2, l2, rinsc3, l3)
    Ltot = l1 + l2 + l3

    #Antall bilder av hver radius:
    nr1 = nCrossCube*l1/Ltot
    nr2 = nCrossCube*l2/Ltot
    nr3 = nCrossCube*l3/Ltot
    
    
    
    nr = [round(nr1), round(nr2), round(nr3)] #Antall bilder av hver cross section. 
    
    if (nr[0] + nr[1] + nr[2]) > 16 :

        nrest1 = nCrossCube*l1/Ltot - nr[0]
        nrest2 = nCrossCube*l2/Ltot - nr[1]
        nrest3 = nCrossCube*l3/Ltot - nr[2]
        n = [nrest1, nrest2, nrest3]
        minst = min(nrest1, nrest2, nrest3)
        ind  = n.index(minst)
        nr[ind] = nr[ind]-1

        
    elif (nr[0] + nr[1] + nr[2]) < 16:

        nrest1 = nCrossCube*l1/Ltot - nr[0]
        nrest2 = nCrossCube*l2/Ltot - nr[1]
        nrest3 = nCrossCube*l3/Ltot - nr[2]
        n = [nrest1, nrest2, nrest3]
        stor = max(nrest1, nrest2, nrest3)
        ind  = n.index(stor)
        nr[ind] = nr[ind]+1

    
    arr1 = np.ones((n_pCube,n_pCube))
    arr2 = np.ones((n_pCube,n_pCube))
    arr3 = np.ones((n_pCube,n_pCube))
    arr = [arr1, arr2, arr3]
    stroke = 0.9
    
    #Regner ut radius i antall piksler
    
    radius1 = rinsc1/Ltot*n_pCube
    radius2 = rinsc2/Ltot*n_pCube
    radius3 = rinsc3/Ltot*n_pCube

    
    rad = [radius1, radius2, radius3]

    
    for i in range(len(rad)):
        inner_radius = rad[i] - (stroke // 1) + (stroke % 1) - 1 
        outer_radius = rad[i] + ((stroke + 1) // 1)
        
        ri, ci = draw.circle(n_pCube/2, n_pCube/2, radius=inner_radius, shape=arr[i].shape)
        ro, co = draw.circle(n_pCube/2, n_pCube/2, radius=outer_radius, shape=arr[i].shape)

        arr[i][ro, co] = 0
        arr[i][ri, ci] = 0
        
        arr =  [arr1, arr2, arr3]

        
    if ver == '3D':
        im3D = []
        for i in range(nr[0]):
            im3D.append(np.concatenate(arr[0]))
            
        for i in range(nr[1]):
            im3D.append(np.concatenate(arr[1]))
            
        for i in range(nr[2]):
            im3D.append(np.concatenate(arr[2])) 
        
        im2D = 0
        im3D = reorg3D('x','x', im3D, nCrossCube)

    elif ver == 'begge':
        
        im2D = []
        for i in range(nr[0]):
            im2D.extend(np.concatenate(arr[0]))
            
        for i in range(nr[1]):
            im2D.extend(np.concatenate(arr[1]))
            
        for i in range(nr[2]):
            im2D.extend(np.concatenate(arr[2])) 
    
        im2D = reorg('x','x', im2D, nCrossCube)
        
        im3D = []
        for i in range(nr[0]):
            im3D.append(np.concatenate(arr[0]))
            
        for i in range(nr[1]):
            im3D.append(np.concatenate(arr[1]))
            
        for i in range(nr[2]):
            im3D.append(np.concatenate(arr[2])) 
       
        im3D = reorg3D('x','x', im3D, nCrossCube)
        
    else:
        im2D = []
        for i in range(nr[0]):
            im2D.extend(np.concatenate(arr[0]))
            
        for i in range(nr[1]):
            im2D.extend(np.concatenate(arr[1]))
            
        for i in range(nr[2]):
            im2D.extend(np.concatenate(arr[2])) 
        im3D = 0
        im2D = reorg('x','x', im2D, nCrossCube)
    

    return im3D, im2D, g123


  
def tilf(start_r, end_r, start_l, end_l): #Genererer semi-tilfeldige tuber
    rad = [random.randint(start_r, end_r)for i in range(3) ]
    l = [random.randint(start_l, end_l) for i in range(3)]
    
    for i in range(len(rad)):
         if rad[i] > sum(l)/2*0.85:
             rad[i] = sum(l)/2*0.85
             
    return rad, l 


def testCircles(Train, Test, Valid, ver):

    start_l = 10
    end_l = 120
    start_r = 1
    end_r = 100
    images2D = []
    images3D = []
    h = []
    nLinks = Train + Test + Valid
    for i in range(nLinks):
        #random radius x 3
        #random lengder x 3
        
        rad, l = tilf(start_r, end_r, start_l, end_l)

        
        im3D, im2D, g = circleHC(rad[0], l[0], rad[1], l[1], rad[2], l[2], ver)

        h.append(g)
        if ver == '2D':
            images2D.append(im2D)
            trainingData2D = images2D[0:Train]
            testData2D = images2D[Train:(Train+Test)]
            validationData2D = images2D[(Train+Test):nLinks]
            
            np.save('codeOutputs/TestCircles/2D/' + str(Train)+'/C2DmatriseTrening' + str(Train)+'.npy', np.array(trainingData2D))
            np.save('codeOutputs/TestCircles/2D/' + str(Train)+'/C2DmatriseValid' + str(Valid)+'.npy', np.array(validationData2D))
            np.save('codeOutputs/TestCircles/2D/' + str(Train)+'/C2DmatriseTest' + str(Test)+'.npy', np.array(testData2D))
            
            trainingHC = h[0:Train]
            testHC = h[Train:(Train+Test)]
            validationHC = h[(Train+Test):nLinks]
            
            np.save('codeOutputs/TestCircles/2D/' + str(Train)+'/CHCTrening' + str(Train)+'.npy', np.array(trainingHC))
            np.save('codeOutputs/TestCircles/2D/' + str(Train)+'/CHCValid' + str(Valid)+'.npy', np.array(validationHC))
            np.save('codeOutputs/TestCircles/2D/' + str(Train)+'/CHCTest' + str(Test)+'.npy', np.array(testHC))
            
            
        
        if ver == '3D':
            images3D.append(im3D)
            trainingData3D = images3D[0:Train]
            testData3D = images3D[Train:(Train+Test)]
            validationData3D = images3D[(Train+Test):nLinks]
   
            
            np.save('codeOutputs/TestCircles/3D/' + str(Train)+'/C3DmatriseTrening' + str(Train)+'.npy', np.array(trainingData3D))
            np.save('codeOutputs/TestCircles/3D/' + str(Train)+'/C3DmatriseValid' + str(Valid)+'.npy', np.array(validationData3D))
            np.save('codeOutputs/TestCircles/3D/' + str(Train)+'/C3DmatriseTest' + str(Test)+'.npy', np.array(testData3D))
            
            trainingHC = h[0:Train]
            testHC = h[Train:(Train+Test)]
            validationHC = h[(Train+Test):nLinks]
            
            np.save('codeOutputs/TestCircles/3D/' + str(Train)+'/CHCTrening' + str(Train)+'.npy', np.array(trainingHC))
            np.save('codeOutputs/TestCircles/3D/' + str(Train)+'/CHCValid' + str(Valid)+'.npy', np.array(validationHC))
            np.save('codeOutputs/TestCircles/3D/' + str(Train)+'/CHCTest' + str(Test)+'.npy', np.array(testHC))
        
          
        if ver == 'begge':
            images2D.append(im2D)
            images3D.append(im3D)
            
            
            trainingData2D = images2D[0:Train]
            testData2D = images2D[Train:(Train+Test)]
            validationData2D = images2D[(Train+Test):nLinks]
            
            
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/C2DmatriseTrening' + str(Train)+'.npy', np.array(trainingData2D))
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/C2DmatriseValid' + str(Valid)+'.npy', np.array(validationData2D))
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/C2DmatriseTest' + str(Test)+'.npy', np.array(testData2D))

              
            
            trainingData3D = images3D[0:Train]
            testData3D = images3D[Train:(Train+Test)]
            validationData3D = images3D[(Train+Test):nLinks]

            
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/C3DmatriseTrening' + str(Train)+'.npy', np.array(trainingData3D))
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/C3DmatriseValid' + str(Valid)+'.npy', np.array(validationData3D))
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/C3DmatriseTest' + str(Test)+'.npy', np.array(testData3D))


            trainingHC = h[0:Train]
            testHC = h[Train:(Train+Test)]
            validationHC = h[(Train+Test):nLinks]
            
            
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/CHCTrening' + str(Train)+'.npy', np.array(trainingHC))
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/CHCValid' + str(Valid)+'.npy', np.array(validationHC))
            np.save('codeOutputs/TestCircles/2D3D/' + str(Train)+'/CHCTest' + str(Test)+'.npy', np.array(testHC))
        
     
    
    return images2D, images3D, h
    
