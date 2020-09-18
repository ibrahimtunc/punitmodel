# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 00:34:14 2020

@author: Ibrahim Alperen Tunc
"""

#Plot the frequency-dependent long delay index for different membrane tau and refractory period values

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
import pandas as pd

savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
datapathes = os.listdir(savepath)
paths = [ path[:5] == 'decay' and path[29]!='1' for path in datapathes]
pathes = list(np.array(datapathes)[paths])

fig, ax = plt.subplots(1,3)

for i, dataname in enumerate(pathes):
    data = pd.read_csv(savepath+'\\'+dataname)
    print(data[:10])
    img = ax[i].imshow(data, extent = [0,data.shape[0],
                                          data.shape[1],0], vmin = 1, vmax = 6)
    ax[i].xaxis.tick_top()
    ax[i].set_xticks(np.arange(0,data.shape[0]+1,10))
    ax[i].set_yticks(np.arange(0,data.shape[0]+1,10))
    ax[i].set_xticklabels(np.round(np.logspace(np.log10(50), np.log10(200), 11),1))
    ax[i].set_yticklabels(np.round(np.logspace(np.log10(50), np.log10(200), 11),1))

    fig.colorbar(img, ax=ax[i], shrink=0.5)
    #plt.gca().invert_yaxis()
    ax[i].set_title('f = %1.f Hz' %(eval(dataname[29:34])), y = 1.05)    
    ax[i].tick_params(axis='both', labelsize=8)
plt.subplots_adjust(left=0.05, right=1, wspace=0.07)
ax[0].set_ylabel('membrane tau [ms]')
ax[0].set_xlabel('refractory period [ms]')


newpaths = [ path[:5] == 'decay' and path[29]=='1' for path in datapathes]
newpathes = list(np.array(datapathes)[newpaths])
fig, ax = plt.subplots(1,4)

for i, dataname in enumerate(newpathes):
    data = pd.read_csv(savepath+'\\'+dataname)
    print(data[:10], np.max(np.max(data)), np.min(np.min(data)))
    img = ax[i].imshow(data, extent = [0,data.shape[0],
                                          data.shape[1],0], vmin = 0, vmax = 13)
    ax[i].xaxis.tick_top()
    ax[i].set_xticks(np.arange(0,data.shape[0]+1,4))
    ax[i].set_yticks(np.arange(0,data.shape[0]+1,4))
    ax[i].set_xticklabels(np.round(np.logspace(np.log10(1), np.log10(1000), 6),1))
    ax[i].set_yticklabels(np.round(np.logspace(np.log10(1), np.log10(1000), 6),1))

    fig.colorbar(img, ax=ax[i], shrink=0.5)
    #plt.gca().invert_yaxis()
    ax[i].set_title('f = %1.f Hz' %(eval(dataname[29:30+i])), y = 1.1)    
    ax[i].tick_params(axis='both', labelsize=8)
plt.subplots_adjust(left=0.05, right=1, wspace=0.1)
ax[0].set_ylabel('membrane tau [ms]')
ax[0].set_xlabel('refractory period [ms]')
