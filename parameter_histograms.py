# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:36:43 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers


random.seed(666)

#Histograms for the parameters over the entire cell populations.

parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

paramarray = np.zeros([len(parameters), len(parameters[0].keys())-1]) #array of parameters

paramnames = list(parameters[0].keys())[1:]

longdecaycellidxs = [8, 10, 13, 15, 16, 17, 24, 26, 29, 31, 34, 38, 45, 59, 61, 66] #indexes of cells showing long decay

for i, param in enumerate(parameters):
    paramarray[i,:] = np.array(list(param.values())[1:])   

   
fig, (*ax) = plt.subplots(3,5, constrained_layout=True, figsize = (12,6))
ax = np.squeeze(ax).reshape(15)

for i, axis in enumerate(ax):
    if i == len(ax)-1:
        continue
    axis.hist(paramarray[:,i], bins=20)
    axis.hist(paramarray[longdecaycellidxs,i], bins=16)
    axis.set_title(paramnames[i])
ax[0].legend(['all', 'long decay'])


#Check pairwise the time scale parameters (mem tau, dend tau, ref_period, tau_a) 6 subplots (2x3)
import itertools
fig, axs = plt.subplots(2, 3)
axs = np.reshape(axs,6)
params = ['dend_tau', 'mem_tau', 'ref_period', 'tau_a']
paramcombs = list(itertools.combinations(params,2))

for i, cell in enumerate(parameters):
    for idx, pair in enumerate(paramcombs):
        if i not in longdecaycellidxs:
            marker = '.k'
        else:
            marker = 'r.'
            
        axs[idx].plot(cell[pair[0]], cell[pair[1]], marker)
        axs[idx].set_title('%s, %s'%(pair))

import matplotlib.lines as mlines
nodecay = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                          markersize=10, label='No decay')
longdecay = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
                          markersize=10, label='Long decay')

axs[-1].legend(handles=[nodecay,longdecay], loc='upper left')
