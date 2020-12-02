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


figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 15,
           'ytick.labelsize' : 15,
           'legend.fontsize' : 15}
plt.style.use(figdict)

#Histograms for the parameters over the entire cell populations.

parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

paramarray = np.zeros([len(parameters), len(parameters[0].keys())-1]) #array of parameters

paramnames = list(parameters[0].keys())[1:]

longdecaycellidxs = [8, 10, 13, 15, 16, 17, 24, 26, 29, 31, 34, 38, 45, 59, 61, 66] #indexes of cells showing long decay

thresholdedlongdecaycellidxs = [13, 16, 26, 29, 34, 59]
for i, param in enumerate(parameters):
    paramarray[i,:] = np.array(list(param.values())[1:])   

   
fig, (*ax) = plt.subplots(3,5, constrained_layout=True, figsize = (12,6))
ax = np.squeeze(ax).reshape(15)
fig.suptitle('Parameter distribution of the p-unit models', size=30)

for i, axis in enumerate(ax):
    if i == len(ax)-1:
        continue
    axis.hist(paramarray[:,i], color='k', label='all', bins=20)
    axis.hist(paramarray[longdecaycellidxs,i], color='r', label='long decay (manual)', bins=16)
    axis.hist(paramarray[thresholdedlongdecaycellidxs,i], color='b', label='long decay (auto)', bins=16)
    axis.set_title(paramnames[i])
ax[-1].legend(*ax[0].get_legend_handles_labels())
ax[-1].axis('off')

#Check pairwise the time scale parameters (mem tau, dend tau, ref_period, tau_a) 6 subplots (2x3)
import itertools
fig, axs = plt.subplots(2, 3)
fig.suptitle('Pairwise distrubution of temporal model parameters', size=30)
axs = np.reshape(axs,6)
params = ['dend_tau', 'mem_tau', 'ref_period', 'tau_a']
paramcombs = list(itertools.combinations(params,2))

for i, cell in enumerate(parameters):
    for idx, pair in enumerate(paramcombs):
        if i not in longdecaycellidxs:
            marker = '.k'
        
        elif i in thresholdedlongdecaycellidxs:
            marker = 'b.'
        
        else:
            marker = 'r.'
            
        axs[idx].plot(cell[pair[0]], cell[pair[1]], marker)
        axs[idx].set_title('%s, %s'%(pair))

import matplotlib.lines as mlines
nodecay = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                          markersize=10, label='No decay')
longdecay = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
                          markersize=10, label='Long decay (manual)')

longdecayauto = mlines.Line2D([], [], color='blue', marker='.', linestyle='None',
                          markersize=10, label='Long decay (auto)')

axs[-1].legend(handles=[nodecay,longdecay, longdecayauto], loc='upper left', prop={'size': 10})
axs[-1].set_xticks(np.linspace(0,0.0015,4))
