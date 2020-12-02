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
white = False
#cell and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
cell_idx = 0
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)

#Stimulus and white noise parameters
whitenoiseparams = {'cflow' : 0, #lower cutoff frequency
                    'cfup' : 300, #upper cutoff frequency
                    'dt' : cellparams['deltat'], #inverse of sampling rate
                    'duration' : 100 #in seconds
                    }

frequency = EODf
contrast = 0.1
fupexample = 10
fAM = 50

locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!
whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
t = np.linspace(0, duration, len(whtnoise))
AMsinewave = contrast * np.sin(2*np.pi*fAM*t)

if white==True:
    stimulus = np.sin(2*np.pi*frequency*t) * (1 + whtnoise)
else: 
     stimulus = np.sin(2*np.pi*frequency*t) * (1 + AMsinewave)
examplestimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*helpers.whitenoise(cflow, fupexample, dt, duration))
fig, axexs = plt.subplots(1,1)
axexs.plot(t, examplestimulus)
axexs.set_title('Example stimulus, $f_{cutoff}=%.2f$'%(fupexample))
axexs.set_xlabel('Time [s]')
axexs.set_ylabel('Amplitude')


#power spectrum
nperseg = 2**12
f, p = welch(stimulus, fs = 1/dt, nperseg=nperseg)

fig, (axs, axp, axpr, axtf) = plt.subplots(1,4)
fig.suptitle(cell)
if white==True:
    fig.text(0.22,0.925,'White noise stimulus', fontsize=15)    
else:
    fig.text(0.22,0.925,'SAM stimulus', fontsize=15)
axs.plot(t[:1000],stimulus[:1000])
axp.plot(f[(f<2*frequency) & (f>0)],
           helpers.decibel_transformer(p[(f<2*frequency) & (f>0)]))  

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


spiketimes = mod.simulate(stimulus, **cellparams)
    
fexample, pexample, meanspkfr = helpers.power_spectrum(whtnoise, spiketimes, t, kernel, nperseg)

axpr.plot(fexample[(fexample<cfup) & (fexample>cflow)],
                   helpers.decibel_transformer(pexample[(fexample<cfup) & (fexample>cflow)])) 
#I do not know if response power spectrum looks correct, but it the model is definitely non-linear because non-existent
#frequencies are pesented in the response (around 1000 Hz)

axpr.set_title('Response power spectrum')
axpr.set_xlabel('Frequency [Hz]')
axpr.set_ylabel('Power [dB]')

fcsd, psr = helpers.cross_spectral_density(whtnoise, spiketimes, t, kernel, nperseg)
fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)

transferfunc = np.abs(psr / pwht) 


axtf.plot(fcsd[fcsd<cfup], transferfunc[fcsd<cfup])
axtf.set_title('Transfer function')
axtf.set_xlabel('Frequency [Hz]')
axtf.set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
#Transfer function values are bit below of the transfer function calculated by SAM stimulus, because of the Parseval 
#theorem. You need to adjust accordingly in the next step (work of a later time!).
fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.89, wspace=0.25, hspace=0)
