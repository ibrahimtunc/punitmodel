# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:23:34 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
from scipy.interpolate import interp1d as interpolate

#Plot the transfer function for the cells with different stimulus contrasts. (also maybe for different amplitudes)
random.seed(666)

#stimulus parameters
tlength = 100
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
fAMs = np.linspace(0,300,31)
fAMs[0]+=1
defcontrast = 0.1
contrasts = np.linspace(0, 1, 21)
amplitudes = np.linspace(0.5, 1.5, 21)
contrasts[0]+= 0.01

parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
 
tfAMscontrast = np.zeros([len(fAMs),len(contrasts)]) #transfer function values (at AM) for different contrasts preallocated

for cell_idx in range(len(parameters)):
    #model parameters
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    
    #rest of stimulus parameters depending on model parameters
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    
    #kernel parameters
    kernelparams = {'sigma' : 0.0001, 'lenfactor' : 5, 'resolution' : t_delta}#kernel is muhc shorter for power spectrum

    #create kernel
    kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

    #power spectrum parameters:
    nperseg = 2**15
    
    fig, (axamp, axc) = plt.subplots(1,2)
    fig.suptitle(cell)
    
    for a_idx, amplitude in enumerate(amplitudes):
        print('amplitude = %.2f' %(amplitude))
        tfAMs = helpers.power_spectrum_transfer_function(frequency, t, defcontrast, fAMs, kernel, nperseg, 
                                                         amp=amplitude, **cellparams)
        tfAMscontrast[:,a_idx] = tfAMs
        axamp.plot(fAMs, tfAMs, '--.', label='%.2f'%(amplitude))
        
    for c_idx, contrast in enumerate(contrasts):
        print('contrast = %.2f' %(contrast))
        tfAMs = helpers.power_spectrum_transfer_function(frequency, t, contrast, fAMs, kernel, nperseg, **cellparams)
        tfAMscontrast[:,c_idx] = tfAMs
        axc.plot(fAMs, tfAMs, '--.', label='%.2f'%(contrast))
    
    axamp.set_xlabel('AM Frequency [Hz]')
    axamp.set_ylabel('Power (contrast normalized)')
    axc.set_title('Transfer function at different contrasts')
    axamp.set_title('Transfer function at different amplitudes')
    axc.legend()
    axamp.legend()
    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break