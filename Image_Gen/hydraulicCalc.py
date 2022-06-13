# -*- coding: utf-8 -*-


import laplacePN



def hydra(node2, link1, link2):
    
    
    g_nodes, cum_rad = laplacePN.calc_g_nodes(node2, 'perm')
    g_throats, cum_throat_radius = laplacePN.calc_g_throats(link1, 'perm')
    g = []
    
    for i in range(len(link2)):
        iNodeA, iNodeB = int(link2[i][1]), int(link2[i][2])
        l_1, l_t, l_2 = link2[i][3], link2[i][5], link2[i][4] ##Length of Pore A, Length of throat, Length of Pore B
        l = l_1 + l_t + l_2
        g.append( l * pow(l_1/(g_nodes[iNodeA-1] ) + l_t/(g_throats[i] )  + l_2/(g_nodes[iNodeB-1])  , -1))

    return g


