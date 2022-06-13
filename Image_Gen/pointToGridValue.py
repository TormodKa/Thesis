# -*- coding: utf-8 -*-

import numpy as np
from cylinderUtils import nodesToPoints
from cubeUtils import nodesToCube
from plotPoints import plotPoints
import matplotlib.pyplot as plt
import properties 


def readGrid(gridfile,dim,bin):
    if bin:
        grid=np.fromfile(gridfile,dtype=np.uint8)
    else:
        grid=np.loadtxt(gridfile)

    grid=np.reshape(grid,dim)
    return grid


def nineToOne(value):
    if value == 9:
        value = 1
    return value

 
    
def posToGridValue(aPos_xyz, grid, arrayY, arrayV, fScale, dim):
    y = [np.floor(aPos_xyz[0]*fScale)-1, np.floor(aPos_xyz[1]*fScale)-1, np.floor(aPos_xyz[2]*fScale)-1]  
    
    if (y[0] < dim[2]) and (y[1] < dim[1]) and (y[2] < dim[0]):
        arrayY.append(y)
        v = grid[int(y[2])][int(y[1])][int(y[0])]
        arrayV.append(nineToOne(v))

    return arrayY, arrayV
    
    

def nodesToGridVal(iNodeA, iNodeB, ax1, ax2, grid, ver = 'cylinder'): #Input Nodenummer, som int, node-koordinater og lest grid
    fScale= properties.fScale
    dim = properties.dim
    
    ######## Sylinder ############
    if ver == 'cylinder':
        nLayers = properties.nLayers
        nCrossSections = properties.nCrossSections
        points = nodesToPoints(ax1, ax2, nCrossSections, nLayers)
    
    ######### Kube ##############
    elif ver == 'cube':
        nCrossSections = properties.nCrossCube
        points, u, v, p1, p2 = nodesToCube(ax1, ax2, nCrossSections)
    
    coordList =[]
    valueList = []

    for point in points:                           
        coordList, valueList = posToGridValue(point, grid, coordList, valueList, fScale, dim)
 
    gx1 = [int(np.floor(ax1[0]*fScale)), int(np.floor(ax1[1]*fScale)), int(np.floor(ax1[2]*fScale))]
    gx2 = [int(np.floor(ax2[0]*fScale)), int(np.floor(ax2[1]*fScale)), int(np.floor(ax2[2]*fScale))]
    return coordList, valueList, ([iNodeA, iNodeB, gx1, gx2, valueList])

