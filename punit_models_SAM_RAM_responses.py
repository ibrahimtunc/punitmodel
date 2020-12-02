# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 23:32:33 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helpers
from scipy.signal import welch
from scipy.interpolate import interp1d as interpolate
import pandas as pd

#SAM RAM responses of the p-unit models for different contrasts (detailed scan)

contrasts = np.linspace(0,0.5,51)
contrasts[0] += 0.001

parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
tlength = 100
correct = False #dont correct SAM and RAM

cflow = 0
cfup = 300
nperseg = 2**12

#SAM parameters
fAMs = np.logspace(np.log10(1),np.log10(300),21)

cell_idx = 0

cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
dt = cellparams['deltat']

#RAM white noise parameters
whitenoiseparams = {'cflow' : cflow, #lower cutoff frequency
                    'cfup' : cfup, #upper cutoff frequency
                    'dt' : dt, #inverse of sampling rate
                    'duration' : tlength #in seconds
                    }
locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!
frequency = EODf
#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt}#kernel is muhc shorter for power spectrum

#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

restparameters = {'fAMS' : fAMs,
                  'cellparams' : cellparams,
                  'whitnoiseparams' : whitenoiseparams,
                  'kernel' : kernel,
                  'nperseg' : nperseg,
                  'frequency' : frequency,
                  'tlength' : tlength,
                  'correct' : correct}

def plot_SAM_RAM_responses(contrasts, **restparameters):
    t = np.arange(0, tlength, dt)

    fRAM, __ = welch(t[1:], nperseg=nperseg, fs=1/dt) #the frequency array for RAM response

    #Run the responses for the given cell parameters
    SAMresponses1, RAMresponses1 = helpers.response_calculator(contrasts, fAMs, cellparams, whitenoiseparams,  
                                                           kernel, nperseg, frequency, tlength, correct=correct)
    #PLotting
    fig, (axsam, axram) = plt.subplots(1,2)
    fig.suptitle('%s model responses to SAM and RAM in different frequencies and v_offsets' %(cell))
    msize = 4
    lwidth = 1
    sammarkers = ['ro-', 'r*-', 'rp-', 'rd-']
    rammarkers = ['ko-', 'k*-', 'kp-', 'kd-']
    samfindex = [9,12,14,19]
    ramfindex = [3,6,11,46]
    

    for idx in range(4):
        axsam.plot(contrasts, SAMresponses1[:, samfindex[idx]], sammarkers[idx], label='f=%.2f' %(fAMs[samfindex[idx]]), 
                   markersize=msize, linewidth=lwidth)
        axsam.set_xlim([0, np.max(contrasts)+0.01])
        
        axram.plot(contrasts, RAMresponses1[:, ramfindex[idx]], rammarkers[idx], label='f=%.2f' %(fRAM[ramfindex[idx]]),
                   markersize=msize, linewidth=lwidth)
        axram.set_xlim([0, np.max(contrasts)+0.01])
                
    axsam.legend()    
    axram.legend()
        
    axsam.set_ylabel('Power')
    fig.text(0.5, 0.05, 'Contrast', size=15)
    axsam.set_title('SAM response')
    axram.set_title('RAM response')
    return axsam, axram
#Run the simulation for different contrasts 
axsam, axram = plot_SAM_RAM_responses(contrasts)
contrasts = np.logspace(-7, -1, 7)
axsam, axram = plot_SAM_RAM_responses(contrasts)
axram.set_xlim()
axsam.set_xlim()

axsam.set_xscale('log')     
axram.set_xscale('log')     

#Check the effect of input scaling and v_offset for a fixed contrast on the transfer function:
#Run the responses for the given cell parameters
contrast = 0.1

fig, axs = plt.subplots(2,2)
axs = np.reshape(axs,4)
axis = axs[0]
axvs = axs[1]
axir = axs[2]
axvr = axs[3]

fig.suptitle('Transfer functions for different model parameters, contrast=%.1f' %(contrast))
axis.set_title('Input scaling')
axvs.set_title('Voltage offset')
axis.set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
axir.set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
axir.set_xlabel('Frequency [Hz]')
axvr.set_xlabel('Frequency [Hz]')

markers = ['o-', '*-', 'p-']

inputscalings = np.array([-5, 0, 5]) + cellparams['input_scaling']
v_offsets = np.array([-5, 0, 5]) + cellparams['v_offset']
whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
t = np.arange(0, tlength, dt)

#RAM stimulus for the model
tRAM = t[1:]
whtstimulus = np.sin(2*np.pi*frequency*tRAM) * (1 + whtnoise)
for idx, inputscaling in enumerate(inputscalings):
    cellparams['input_scaling'] = inputscaling
    
    #RAM stimulus
    whtspiketimes = mod.simulate(whtstimulus, **cellparams)
    #cross spectral density and the transfer function for the RAM
    fcsdRAM, psrRAM = helpers.cross_spectral_density(whtnoise, whtspiketimes, tRAM, 
                                                     kernel, nperseg, calcoherence=False)
    whttransferfunc = np.abs(psrRAM / (pwht))
    whtnoisefrange = [(fcsdRAM>0) & (fcsdRAM<300)]
    
    #SAM stimulus
    pfAMs = np.zeros(len(fAMs)) #power at fAM for stimulus
    pfAMr = np.zeros(len(fAMs)) #power at fAM for stimulus

    for findex, fAM in enumerate(fAMs):
        #print(findex)
        #create stimulus and calculate power at fAM for rectified stimulus
        SAMsinewave = contrast*np.sin(2*np.pi*fAM*t)
        SAMstimulus = np.sin(2*np.pi*frequency*t) * (1 + SAMsinewave)
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        fSAM, pSAM = welch(SAMsinewave, fs=1/dt, nperseg=npersegfAM) #power of the AM sine wave!
        pSAM_interpolator = interpolate(fSAM, pSAM)
        pfAMs[findex] = pSAM_interpolator(fAM)
        SAMspiketimes = mod.simulate(SAMstimulus, **cellparams)
        frSAM, prSAM, __ = helpers.power_spectrum(SAMstimulus, SAMspiketimes, t, kernel, npersegfAM)
        
        #interpolate the response power at fAM, later to be used for the transfer function
        presp_interpolator = interpolate(frSAM, prSAM)
        pfAMr[findex] = presp_interpolator(fAM)
    
    SAMtransferfunc = np.sqrt(pfAMr/pfAMs)
    
    axis.plot(fAMs, SAMtransferfunc, 'r'+markers[idx], label='%.3f' %(inputscaling))
    axir.plot(fcsdRAM[whtnoisefrange], whttransferfunc[whtnoisefrange], 'k'+markers[idx], label='%.3f' %(inputscaling))
    
for idx, v_offset in enumerate(v_offsets):
    cellparams['v_offset'] = v_offset
    
    #RAM stimulus
    whtspiketimes = mod.simulate(whtstimulus, **cellparams)
    #cross spectral density and the transfer function for the RAM
    fcsdRAM, psrRAM = helpers.cross_spectral_density(whtnoise, whtspiketimes, tRAM, 
                                                     kernel, nperseg, calcoherence=False)
    whttransferfunc = np.abs(psrRAM / (pwht))
    whtnoisefrange = (fcsdRAM>0) & (fcsdRAM<300)
    
    #SAM stimulus
    pfAMs = np.zeros(len(fAMs)) #power at fAM for stimulus
    pfAMr = np.zeros(len(fAMs)) #power at fAM for stimulus

    for findex, fAM in enumerate(fAMs):
        #print(findex)
        #create stimulus and calculate power at fAM for rectified stimulus
        SAMsinewave = contrast*np.sin(2*np.pi*fAM*t)
        SAMstimulus = np.sin(2*np.pi*frequency*t) * (1 + SAMsinewave)
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        fSAM, pSAM = welch(SAMsinewave, fs=1/dt, nperseg=npersegfAM) #power of the AM sine wave!
        pSAM_interpolator = interpolate(fSAM, pSAM)
        pfAMs[findex] = pSAM_interpolator(fAM)
        SAMspiketimes = mod.simulate(SAMstimulus, **cellparams)
        frSAM, prSAM, __ = helpers.power_spectrum(SAMstimulus, SAMspiketimes, t, kernel, npersegfAM)
        
        #interpolate the response power at fAM, later to be used for the transfer function
        presp_interpolator = interpolate(frSAM, prSAM)
        pfAMr[findex] = presp_interpolator(fAM)
    
    SAMtransferfunc = np.sqrt(pfAMr/pfAMs)
    
    axvs.plot(fAMs, SAMtransferfunc, 'r'+markers[idx], label='%.3f' %(v_offset))
    axvr.plot(fcsdRAM[whtnoisefrange], whttransferfunc[whtnoisefrange], 'k'+markers[idx], label='%.3f' %(inputscaling))

for ax in axs:
    ax.legend()
