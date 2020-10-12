# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:28:20 2020

@author: Ibrahim Alperen Tunc
"""
import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
from scipy.signal import welch, csd
from scipy.interpolate import interp1d as interpolate
import os 
import pandas as pd

#Power spectrum with white noise

#cell and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
cell_idx = 0
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)

#White noise parameters
whitenoiseparams = {'cflow' : 0, #lower cutoff frequency
                    'cfup' : 300, #upper cutoff frequency
                    'dt' : cellparams['deltat'], #inverse of sampling rate
                    'duration' : 100 #in seconds
                    }

locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!

stimulus = helpers.whitenoise(**whitenoiseparams)
t = np.linspace(0, duration, len(stimulus))

#power spectrum
nperseg = 2**15
f, p = welch(stimulus, fs = 1/dt, nperseg=nperseg)

fig, (axs, axp, axpr, axtf) = plt.subplots(1,4)
fig.suptitle(cell)
fig.text(0.22,0.925,'White noise', fontsize=15)    


axs.plot(t,stimulus)
axp.plot(f[(f<cfup) & (f>cflow)],helpers.decibel_transformer(p[(f<cfup) & (f>cflow)])) #dont mind the undefined names,
                                                                                         #they are defined by locals line
axs.set_title('Time Domain')
axs.set_xlabel('Time [s]')
axs.set_ylabel('Amplitude')
axp.set_title('Frequency Domain')
axp.set_xlabel('Frequency [Hz]')
axp.set_ylabel('Power [dB]')

#Ok white noise works well, you understood whats up. Now get the transfer function by using the model response
    
#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt}#kernel is muhc shorter for power spectrum

#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

#power spectrum parameters:
nperseg = 2**15

spiketimes = mod.simulate(stimulus, **cellparams)
    
fexample, pexample, meanspkfr = helpers.power_spectrum(stimulus, spiketimes, t, kernel, nperseg)

axpr.plot(fexample[(fexample<cfup) & (fexample>cflow)],
                   helpers.decibel_transformer(pexample[(fexample<cfup) & (fexample>cflow)])) 
#I do not know if response power spectrum looks correct, but it the model is definitely non-linear because non-existent
#frequencies are pesented in the response (around 1000 Hz)

axpr.set_title('Response power spectrum')
axpr.set_xlabel('Frequency [Hz]')
axpr.set_ylabel('Power [dB]')

fcsd, psr = helpers.cross_spectral_density(stimulus, spiketimes, t, kernel, nperseg)

transferfunc = np.real(psr / p) 

axtf.plot(fcsd[fcsd<cfup], transferfunc[fcsd<cfup])
axtf.set_title('Transfer function')
axtf.set_xlabel('Frequency [Hz]')
axtf.set_ylabel('Power')
fig.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.89, wspace=0.25, hspace=0)
