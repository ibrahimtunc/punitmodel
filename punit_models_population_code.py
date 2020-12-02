# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:49:02 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helpers
from scipy.signal import coherence
from matplotlib.ticker import ScalarFormatter
#Create the population code by using the p-unit integrate and fire model neurons.

parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

#create the homogeneous population by randomly choosing one neuron
neuronidx = 5 #chosen by looking at population_heterogeneity.py, cell idx 5 has nice fI curves

#create RAM stimulus
contrast = 0.1 #i will decide soon on the contrast value, this script is initial and therefore works 
               #rather as play field
maxpop = 2**4
                        
tlength = 1

cflow = 0
cfup = 300

cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(neuronidx, parameters)
dt = cellparams['deltat']
nperseg = 2**13
#RAM white noise parameters
whitenoiseparams = {'cflow' : cflow, #lower cutoff frequency
                    'cfup' : cfup, #upper cutoff frequency
                    'dt' : dt, #inverse of sampling rate
                    'duration' : tlength #in seconds
                    }
locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!

t = np.arange(0, tlength, dt)

whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
#RAM stimulus for the model
tRAM = t[1:]
whtstimulus = np.sin(2*np.pi*EODf*tRAM) * (1 + whtnoise)

#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt}#kernel is muhc shorter for power spectrum

#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
I_LBshomo = np.zeros(npops.shape)
I_LBshetero = np.zeros(npops.shape)

for idx, npop in enumerate(npops):
    npop += 1
    print(idx)
    #homogeneous population
    popacthomo = helpers.homogeneous_population(npop, tRAM, whtstimulus, cellparams, kernel)
    I_LBhomo = helpers.lower_bound_info(summedactconvhomo, whtstimulus, tRAM, nperseg, cflow, cfup)    
    I_LBshomo[idx] = I_LBhomo

    #heterogeneous population
    summedactconvhetero = [0]
    while np.max(summedactconvhetero) == 0: #rerun the population simulation until the chosen population can respond
                                            #to the given population.
        popacthetero, summedactconvhetero = helpers.heterogeneous_population(npop, tRAM, whtstimulus, kernel)
        if np.max(summedactconvhetero) == 0:
            print('Stimulus cannot drive the population!')
    I_LBhetero = helpers.lower_bound_info(summedactconvhetero, whtstimulus, tRAM, nperseg, cflow, cfup)    
    I_LBshetero[idx] = I_LBhetero

fig, ax = plt.subplots(1,1)
ax.plot(npops, I_LBshomo, label='Homo population')
ax.plot(npops, I_LBshetero, label='Hetero population')
ax.set_xscale('log',basex=2) 
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_ylabel('$I_{LB}$')
ax.set_xlabel('Population size')
ax.set_xticks(np.logspace(np.log10(2), np.log10(maxpop), np.log2(maxpop).astype(int)))
ax.legend()

