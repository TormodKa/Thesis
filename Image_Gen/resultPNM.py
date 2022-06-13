# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:23:36 2022

@author: lavra
"""

import numpy as np
from laplacePN import calc_flow
from netRecon import get_network_file


node1 = get_network_file('node1')
node2 = get_network_file('node2')
link1 = get_network_file('link1')
link2 = get_network_file('link2')
print(len(link1), len(link2), len(node1), len(node2))


###################### Pore Network Model #######################
# afHC = np.load('codeOutputs/TrainTestValid/2D3D/FullSet/FullSetHC.npy', allow_pickle = True)

# q_vec, perm = calc_flow(node1, node2, link1, link2)
# # q_vec = calc_flow_custom_nodes(node1_mod, node2_mod, link1_mod, link2_mod, afHC)
# print('Flowrate: ', q_vec, 'm^3/s')
# np.save('TEST_q_vec_PNM_modified_Nodes&Links.npy', q_vec)
# np.save('TEST_perm_PNM_modified_Nodes&Links.npy', perm)

###################### CNN Model  ################################
# afHC = PREKSJONENE TIL MODELLEN PÅ ALLE BILDENE
q_vec, perm = calc_flow(node1, node2, link1, link2)
print('Flowrate: ', q_vec, 'm^3/s')
np.save('q_vec_PNM_CNN_Nodes&Links.npy', q_vec)
np.save('perm_PNM_CNN_Nodes&Links.npy', perm)
