# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:15:07 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
from scipy.signal import welch
from scipy.interpolate import interp1d as interpolate
import os 
import pandas as pd
import matplotlib as mpl

#Compare transfer function of SAM and RAM stimuli.

#General parameters for both stimuli types
#cell and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
cell_idx = 0
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
dt = cellparams['deltat']
contrasts = np.linspace(0,0.5,11)
contrasts[0] += 0.01
tlength = 100
frequency = EODf
t = np.arange(0, tlength, dt)
cflow = 0
cfup = 300
#SAM parameters
fAMs = np.linspace(0,300,21)
fAMs[0] += 1

#RAM white noise parameters
whitenoiseparams = {'cflow' : cflow, #lower cutoff frequency
                    'cfup' : cfup, #upper cutoff frequency
                    'dt' : dt, #inverse of sampling rate
                    'duration' : 100 #in seconds
                    }
locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!

#Calculate the stimuli
whtnoises = np.zeros([len(t)-1,len(contrasts)])
whtnoisespwr = []
SAMstimpwr = []
nperseg = 2**15
RAMtransferfuncs = []
RAMcoherences = []
SAMtransferfuncs = []

#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt}#kernel is muhc shorter for power spectrum

#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

for cidx, contrast in enumerate(contrasts):
    print(cidx)
    #create white noise for different contrasts
    whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
    whtnoises[:,cidx] = whtnoise
    #calculate white noise power for different contrasts
    fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
    whtnoisespwr.append(pwht)
    
    #RAM stimulus for the model
    tRAM = t[1:]
    whtstimulus = np.sin(2*np.pi*frequency*tRAM) * (1 + whtnoise)
    #model response to RAM stimulus   
    whtspiketimes = mod.simulate(whtstimulus, **cellparams)
    
    #cross spectral density and the transfer function for the RAM
    fcsdRAM, psrRAM, fcohRAM, gammaRAM = helpers.cross_spectral_density(whtnoise, whtspiketimes, tRAM, 
                                                                        kernel, nperseg, calcoherence=True)
    whttransferfunc = np.abs(psrRAM / pwht) 
    RAMtransferfuncs.append(whttransferfunc)
    RAMcoherences.append(gammaRAM)
    
    #same thing as RAM for the SAM at different contrasts, except coherence thing is for now missing.
    #calculate for the given contrast each fAM stimulus and corresponding power
    pfAMs = np.zeros(len(fAMs)) #power at fAM for stimulus
    pfAMr = np.zeros(len(fAMs)) #power at fAM for response
    for findex, fAM in enumerate(fAMs):
        #print(findex)
        #create stimulus and calculate power at fAM for rectified stimulus
        SAMstimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*fAM*t))
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        fSAM, pSAM = welch(np.abs(SAMstimulus-np.mean(SAMstimulus)), fs=1/dt, nperseg=npersegfAM)
        pSAM_interpolator = interpolate(fSAM, pSAM)
        pfAMs[findex] = pSAM_interpolator(fAM)
        
        #model response to the SAM stimulus and power spectrum
        SAMspiketimes = mod.simulate(SAMstimulus, **cellparams)
        frSAM, prSAM, __ = helpers.power_spectrum(SAMstimulus, SAMspiketimes, t, kernel, npersegfAM)
        
        #interpolate the response power at fAM, later to be used for the transfer function
        presp_interpolator = interpolate(frSAM, prSAM)
        pfAMr[findex] = presp_interpolator(fAM)

    SAMstimpwr.append(pfAMs)
    SAMtransferfuncs.append(np.sqrt(pfAMr/pfAMs))
    
whtnoisespwr = np.array(whtnoisespwr)
SAMstimpwr = np.array(SAMstimpwr)
RAMtransferfuncs = np.array(RAMtransferfuncs)
RAMcoherences = np.array(RAMcoherences)
SAMtransferfuncs = np.array(SAMtransferfuncs)

fig, axts = plt.subplots(3,4, sharex=True, sharey='row')
fig.suptitle('SAM and RAM transfer functions at different contrasts, cell %s' %(cell))
lastax = axts[-1,-1]

#remove last ax from sharey
shay = lastax.get_shared_y_axes()
shay.remove(lastax)
#create new yticks for lastax
yticker = mpl.axis.Ticker()
lastax.yaxis.major = yticker
# The new ticker needs new locator and formatters
yloc = mpl.ticker.AutoLocator()
yfmt = mpl.ticker.ScalarFormatter()
lastax.yaxis.set_major_locator(yloc)
lastax.yaxis.set_major_formatter(yfmt)

axts = np.delete(axts.reshape(12), 11)
whtnoisefrange = (fwht>cflow) & (fwht<cfup) #frequency range to plot the power for white nose
for idx, ax in enumerate(axts):
    ax.plot(fcsdRAM[whtnoisefrange], RAMtransferfuncs[idx, :][whtnoisefrange], 'k--', label='RAM')
    ax.plot(fAMs, SAMtransferfuncs[idx,:], 'r.-', label='SAM')
    ax.set_title('contrast=%.2f' %(contrasts[idx]))
    lastax.plot(fcohRAM[whtnoisefrange],RAMcoherences[idx,:][whtnoisefrange])
    lastax.set_ylim([0, 1.0])
axts[4].set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
fig.text(0.45, 0.05, 'Frequency [Hz]')
axts[-1].legend(loc='best')
lastaxyticks = np.linspace(0,1.1,12)
lastax.set_yticks(lastaxyticks)
plt.subplots_adjust(wspace=0.3)
#about to do something super lame:
for idx,tick in enumerate(lastaxyticks):
    fig.text(0.722, 0.105+0.02*idx, np.round(tick,4))
lastax.set_title('RAM coherences')
lastax.set_ylabel('Coherence factor $\gamma$')
lastax.yaxis.set_label_coords(-0.15, 0.5)
#Coherence increases with increase in contrast, meaning that for contrasts until 0.5, more contrast makes the system
#either more linear or less noisy. 