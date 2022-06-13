# -*- coding: utf-8 -*-


import numpy as np
import math
import sympy
from sympy import symbols, Eq, solve
from properties import ratioCyl, n_pCyl, nCrossSections, nLayers, dim, cellSize


def orthogonalPoints(ax1, ax2, ratioCyl, normv1):
    x2, x1, y2, y1, z2, z1 = ax2[0], ax1[0], ax2[1], ax1[1], ax2[2], ax1[2]
    
    
    
    if (x2 == x1) and (y2==y1):
    
        x, z = sympy.symbols('x, z')
        eq1 = sympy.Eq((x2-x1)*(x-x2)+(z2-z1)*(z-z2),0)
        eq2 = sympy.Eq(ratioCyl*normv1-sympy.sqrt(pow((x-x2),2)+pow((z-z2),2)),0)
        sol_dict = sympy.solve((eq1, eq2), (x, z))
        p = np.array([[sol_dict[0][0], y1, sol_dict[0][1]],[sol_dict[1][0], y2, sol_dict[1][1]]]) 
    elif (x2 == x1) and (z2==z1):
    
        y, z = sympy.symbols('y, z')

        eq1 = sympy.Eq((y2-y1)*(y-y2)+(z2-z1)*(z-z2),0) # z1 og z3 er lik, og z-leddet forsvinner
        eq2 = sympy.Eq(ratioCyl*normv1-sympy.sqrt(pow((y-y2),2)+pow((z-z2),2)),0)
        sol_dict = sympy.solve((eq1, eq2), (y, z))
        p = np.array([[x1, sol_dict[0][0], sol_dict[0][1]],[x2, sol_dict[1][0], sol_dict[1][1]]]) 
        
    else:
        x, y = sympy.symbols('x, y')
    
        eq1 = sympy.Eq((x2-x1)*(x-x2)+(y2-y1)*(y-y2),0) # z1 og z3 er lik, og z-leddet forsvinner
        eq2 = sympy.Eq(ratioCyl*normv1-sympy.sqrt(pow((x-x2),2)+pow((y-y2),2)),0)
        sol_dict = sympy.solve((eq1, eq2), (x, y))
        p = np.array([[sol_dict[0][0], sol_dict[0][1], z1],[sol_dict[1][0], sol_dict[1][1],z2]]) 
    return p[0], p[1]


def rotationMatrix(rot_vec,angle_rad, axis_vec): 
    #takes in vector to be rotated by a given angle around an axis-vector
    norm = np.linalg.norm(axis_vec) # calculates the norm of the axis-vector
    #Finds the values of the unit vector
    
    l = axis_vec[0]/norm
    m = axis_vec[1]/norm
    n = axis_vec[2]/norm
    
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    aarot = np.array([
        [l*l*(1 - c) + c, m*l*(1-c) - n*s, n*l*(1-c) + m*s
        ],
        [l*m*(1-c) + n*s, m*m*(1-c) + c, n*m*(1-c) - l*s
        ], 
        [l*n*(1-c) - m*s, m*n*(1-c) + l*s, n*n*(1-c) + c]
        ])
    
    return aarot@rot_vec #returns rotated vector 






def outerToInner(v2, n_pCyl, axis_vec, x2, y2, z2, points, nCrossSections):
    j = 0
    n_points = n_pCyl[j]
    n_v2 = np.array([(nLayers-j)/nLayers*v2[0], (nLayers-j)/(nLayers)*v2[1], ((nLayers)-j)/(nLayers)*v2[2]])
    
    vec = []
    i = 0
    while  j < (nLayers):
        
        vec.append(rotationMatrix(n_v2, math.radians(360/n_points*i), axis_vec))
        q = np.array([(vec[-1][0]+x2), (vec[-1][1]+y2), (vec[-1][2]+z2)])   
        points.append(q)
        i += 1
        
        
        if i == n_points:
            i = 0
            j += 1
        if i ==0 and j < len(n_pCyl):
            n_v2 = np.array([(nLayers-j)/nLayers*v2[0], (nLayers-j)/nLayers*v2[1], (nLayers-j)/nLayers*v2[2]])
            n_points = n_pCyl[j]
            vec = []
        
    points.append(np.array([x2, y2, z2]))
    return points
    



def lastCircle(v2, n_pCyl, axis_vec, x1,y1,z1, points, nCrossSections):
    j = 0
    n_points = n_pCyl[j]
    n_v2 = np.array([(len(n_pCyl)-j)/len(n_pCyl)*v2[0], (len(n_pCyl)-j)/len(n_pCyl)*v2[1], (len(n_pCyl)-j)/len(n_pCyl)*v2[2]])

    vec = []
    i = 0
    while  j < nLayers:
        
        vec.append(rotationMatrix(n_v2, math.radians(360/n_points*i), axis_vec))
        q = np.array([(vec[-1][0]+x1), (vec[-1][1]+y1), (vec[-1][2]+z1)])   
        points.append(q)
        i += 1
        
        
        if i == n_points:
            i = 0
            j += 1
        if i ==0 and j < len(n_pCyl):
            n_v2 = np.array([(nLayers-j)/nLayers*v2[0], (nLayers-j)/nLayers*v2[1], (nLayers-j)/nLayers*v2[2]])
            n_points = n_pCyl[j]
            vec = []
    points.append(np.array([x1,y1,z1]))
    return points


    
    
def closePoint(arrayPoints, x, y, z):
    arrayPoints = np.array(arrayPoints)
    distance = (arrayPoints[:, 0]-x)**2 + (arrayPoints[:, 1]-y)**2 + (arrayPoints[:, 2]-z)**2
    
    return arrayPoints[distance.argmin()]

      
 
def crossSectionOrganizer(a_atm, points, n_pCyl):
    i = 0
    while i <  sum(n_pCyl) :
        
        near = closePoint(a_atm, points[-1][0], points[-1][1], points[-1][2])
        
        points.append(np.array(near))
      
        for j in range(0,len(a_atm)):
            if (a_atm[j][0] == near[0] and a_atm[j][1] == near[1] and a_atm[j][2] == near[2]):
                del a_atm[j]
                break
        i += 1
    return points
        


def pointStructurer(arrayPoints, n_pCyl, nCrossSections):
    
    
    a = arrayPoints
    splitedSize = int(len(a)/nCrossSections)
    a_split = [a[x:x+splitedSize] for x in range(0, len(a), splitedSize)]
    
    points = []
    splitNr = 0
   

    a_atm = a_split[splitNr]
    start = a_atm[-1]
   
    while splitNr < len(a_split):
        
        a_atm = a_split[splitNr]
        points.append(a_atm[-1])
        del a_atm[-1]
        
        newCross = crossSectionOrganizer(a_atm, points, n_pCyl)
        
        splitNr += 1
        
    return points

 


def nodesToPoints(ax1, ax2, nCrossSections, nLayers):   # ax1 - array of coordinates, x, y, and z, for a pore center. 
    ax1 = np.array(ax1)
    ax2 = np.array(ax2)
    
    antall_v2 = nLayers
    antall_v1 = nCrossSections - 1 # Tverrsnittet rundt legges paa uansett

    v1 = ax2-ax1
    axis_vec = v1 #Choosing v1 as the axis to rotate around

    x2, x1, y2, y1, z2, z1 = ax2[0], ax1[0], ax2[1], ax1[1], ax2[2], ax1[2]


    points = []

    
    normv1 = np.linalg.norm(v1) #lengden av vektoren mellom punktene
    normv2 = normv1*ratioCyl     #lengden av vektoren v2 

    


    ### Calculating points

    for k in range(antall_v1, 0, -1):
        x2, y2, z2 = (k/antall_v1)*v1[0]+ax1[0], (k/antall_v1)*v1[1]+ax1[1], (k/antall_v1)*v1[2]+ax1[2]
        p1, p2 = orthogonalPoints(ax1, [x2, y2, z2], ratioCyl, normv1)
        v2 = np.array([float(p2[0]-x2), float(p2[1]-y2), float(p2[2]-z2)])
        points = outerToInner(v2, n_pCyl, axis_vec, x2, y2, z2, points, nCrossSections)


    lastCircle(v2, n_pCyl, axis_vec, x1, y1, z1, points, nCrossSections) #Punktene i rommet.
    
    points = pointStructurer(points, n_pCyl, nCrossSections)
    return points


