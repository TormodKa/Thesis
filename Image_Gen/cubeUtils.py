# -*- coding: utf-8 -*-

import numpy as np
import math
import sympy
from sympy import symbols, Eq, solve
from plotPoints import plotPoints
import matplotlib.pyplot as plt
import time
from properties import ratioCube, n_pCube, nCrossCube, dim, cellSize

def orthogonalPoints(ax1, ax2, normv1):
    x2, x1, y2, y1, z2, z1 = ax2[0], ax1[0], ax2[1], ax1[1], ax2[2], ax1[2]
    ratio = 1/2
    
    
    if (x2 == x1) and (y2==y1):
        x, z = sympy.symbols('x, z')
        eq1 = sympy.Eq((x2-x1)*(x-x2)+(z2-z1)*(z-z2),0)
        eq2 = sympy.Eq(ratioCube*normv1-sympy.sqrt(pow((x-x2),2)+pow((z-z2),2)),0)
        sol_dict = sympy.solve((eq1, eq2), (x, z))
        p = np.array([[sol_dict[0][0], y1, sol_dict[0][1]],[sol_dict[1][0], y2, sol_dict[1][1]]]) 
    elif (x2 == x1) and (z2==z1):
        y, z = sympy.symbols('y, z')
        eq1 = sympy.Eq((y2-y1)*(y-y2)+(z2-z1)*(z-z2),0) # z1 og z3 er lik, og z-leddet forsvinner
        eq2 = sympy.Eq(ratioCube*normv1-sympy.sqrt(pow((y-y2),2)+pow((z-z2),2)),0)
        sol_dict = sympy.solve((eq1, eq2), (y, z))
        p = np.array([[x1, sol_dict[0][0], sol_dict[0][1]],[x2, sol_dict[1][0], sol_dict[1][1]]]) 
    else:
        x, y = sympy.symbols('x, y')
        eq1 = sympy.Eq((x2-x1)*(x-x2)+(y2-y1)*(y-y2),0) # z1 og z3 er lik, og z-leddet forsvinner
        eq2 = sympy.Eq(ratioCube*normv1-sympy.sqrt(pow((x-x2),2)+pow((y-y2),2)),0)
        sol_dict = sympy.solve((eq1, eq2), (x, y))
        p = np.array([[sol_dict[0][0], sol_dict[0][1], z1],[sol_dict[1][0], sol_dict[1][1],z2]]) 
    return p[0], p[1]




def nodesToCube(ax1, ax2, nCrossCube):   # ax1 - array of coordinates, x, y, and z, for a pore center. 
    
    u = [ax2[0]-ax1[0], ax2[1]-ax1[1], ax2[2]-ax1[2]]
    
    x2, x1, y2, y1, z2, z1 = ax2[0], ax1[0], ax2[1], ax1[1], ax2[2], ax1[2]
    nCrossSections = nCrossCube
    points = []
    
    normu = np.linalg.norm(u) #lengden av vektoren mellom punktene
    
    n_p = n_pCube

    nCrossSections -= 1

    p1, p2 = orthogonalPoints(ax1, [x2, y2, z2], normu)
    
    v = [float(p2[0]-ax2[0]), float(p2[1]-ax2[1]), float(p2[2]-ax2[2])]
    w = np.cross(v, u)
    
    ldw = np.linalg.norm(w)/np.linalg.norm(u)
    w = w/ldw/2
    
    cross = nCrossSections
   
    
    i = 0
    j = 0
    s = [(cross/nCrossSections)*ax2[0], (cross/nCrossSections)*ax2[1], (cross/nCrossSections)*ax2[2]]
    punktOV = [w[0]+v[0]+s[0], w[1]+v[1]+s[1], w[2]+v[2]+s[2]]

    
    alpha = [2*v[0]/(n_p-1), 2*v[1]/(n_p-1), 2*v[2]/(n_p-1)]
    beta = [2*w[0]/(n_p-1), 2*w[1]/(n_p-1), 2*w[2]/(n_p-1)]
    while cross >= 0:
        t = [punktOV[0] - i*alpha[0] - j*beta[0], punktOV[1] - i*alpha[1] - j*beta[1], punktOV[2] - i*alpha[2] - j*beta[2] ]
        points.append(t)
        i += 1
        if i == n_p and j == n_p-1:
            i = 0
            j = 0
            cross -= 1
            s = [(cross/nCrossSections)*u[0]+ax1[0], (cross/nCrossSections)*u[1]+ax1[1], (cross/nCrossSections)*u[2]+ax1[2]]
            punktOV = [w[0]+v[0]+s[0], w[1]+v[1]+s[1], w[2]+v[2]+s[2]]
        
        elif i == n_p :
            i = 0
            j += 1
            

            
    toc = time.perf_counter()

    return points, w, v, p1, p2


