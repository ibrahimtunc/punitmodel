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

paramarray = np.zeros([len(parameters), len(parameters[0].keys())-1])

paramnames = list(parameters[0].keys())[1:]

for i, param in enumerate(parameters):
    paramarray[i,:] = np.array(list(param.values())[1:])   

   
fig, (*ax) = plt.subplots(3,5, constrained_layout=True, figsize = (12,6))
ax = np.squeeze(ax).reshape(15)

for i, axis in enumerate(ax):
    if i == len(ax)-1:
        continue
    axis.hist(paramarray[:,i], bins=20)
    axis.set_title(paramnames[i])
