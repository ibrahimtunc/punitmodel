# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:41:12 2020

@author: Ibrahim Alperen Tunc
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
import pandas as pd

savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
datapathes = os.listdir(savepath)
paths = [ path[:5] == 'decay' and len(path) > 100 for path in datapathes]
pathes = list(np.array(datapathes)[paths])
pathes = np.array(pathes)[[0, 1, 8, 4, 3, 5, 2, 6, 7]] #rearrange for nice plot matrixing


fig, axs = plt.subplots(3,3)
axs = np.reshape(axs, 9)
fig.suptitle('Decay index of the reduced LIF model for different parameters, 10 Hz Stimulus frequency', size=20)

for i, dataname in enumerate(pathes):
    data = pd.read_csv(savepath+'\\'+dataname)
    print(data[:10], np.max(np.max(data)), np.min(np.min(data)))
    img = axs[i].imshow(data, extent = [0,data.shape[0],
                                          data.shape[1],0], vmin = 0, vmax = 13)
    axs[i].xaxis.tick_top()
    axs[i].set_xticks(np.linspace(0,data.shape[0],4))
    axs[i].set_yticks(np.linspace(0,data.shape[0],4))
    axs[i].set_xticklabels(np.round(np.logspace(np.log10(1), np.log10(1000), 4),1))
    axs[i].set_yticklabels(np.round(np.logspace(np.log10(1), np.log10(1000), 4),1))

    fig.colorbar(img, ax=axs[i], shrink=0.5)
    
    axs[i].set_title('$I_{off}$ = %d, dt = %.6f, amp = %d,' %(eval(dataname[40:45]), eval(dataname[56:64]),
       eval(dataname[73:80])), y = 1.1)    

    axs[i].tick_params(axis='both', labelsize=8)
plt.subplots_adjust(left=0.05, right=1, wspace=0.1)
axs[3].set_ylabel('membrane tau [ms]', size=20)
axs[7].set_xlabel('refractory period [ms]', size=20)