# -*- coding: utf-8 -*-


ratioCyl = 1/4 #forholdstall mellom lengde v2 og v1
ratioCube = 1/2 #Forholdstall mellom v2 og v1

# n_pCyl = [49, 36, 25, 16, 9, 4 ]
# n_pCyl = [16, 16, 16, 16, 16, 16]
# n_pCyl = [40, 36, 32, 28, 24, 20, 16, 12, 8]
# n_pCyl = [64]*16
# n_pCube = 15
n_pCube = 16  ## Pixler n_pCube x n_pCube: 32x32

nCrossSections = 16
nLayers = 16

nCrossCube = 16
dim = [1180, 764, 792]
cellSize = 6.0772E-06 
fScale = 1/cellSize

g = "grid_MM_792x764x1180.raw"

padd = False
################## ENDRE UTELUKKENDE PADDSIZE ##################
paddSize = 12 

topnbottom = [1]*(n_pCube+paddSize)*int(paddSize/2)
side = [1]*int(paddSize/2)

