# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:09:23 2020

@author: Ibrahim Alperen Tunc
"""

import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helpers
from scipy.signal import welch
from scipy.interpolate import interp1d as interpolate
import os 
import pandas as pd

#Check the power of the SAM stimuli alltogether as well as the one of the AM sine wave
#SAM parameters
fAMs = np.logspace(np.log10(100),np.log10(400),21)
pfAMs = np.zeros(len(fAMs)) #power at fAM for stimulus
pfAMAM = np.zeros(len(fAMs)) #power at fAM for AM
EODf = 800
tlength = 100
dt = 5 * 10**-5
t = np.arange(0, tlength, dt)
contrasts = np.linspace(0,0.5,11)
contrasts[0] += 0.01

#RAM parameters
cflow = 99
cfup = 401

#RAM white noise parameters
whitenoiseparams = {'cflow' : cflow, #lower cutoff frequency
                    'cfup' : cfup, #upper cutoff frequency
                    'dt' : dt, #inverse of sampling rate
                    'duration' : 100 #in seconds
                    }
locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!
whtnoises = np.zeros([len(t)-1,len(contrasts)])
whtnoisespwr = []
whtnoiseAMpwr = []
nperseg = 2**12

for contrast in contrasts:
    whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
    fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
    whtnoisefrange = (fwht>cflow) & (fwht<cfup) #frequency range to plot the power for white nose
    whtnoiseAMpwr.append(pwht[whtnoisefrange])
    
    tRAM = t[1:]
    whtstimulus = np.sin(2*np.pi*EODf*tRAM) * (1 + whtnoise)
    fwhtstim, pwhtstim = welch(np.abs(whtstimulus-np.mean(whtstimulus)), fs=1/dt, nperseg=nperseg)
    whtstimfrange = (fwhtstim>cflow) & (fwhtstim<cfup) #frequency range to plot the power for white nose
    whtnoisespwr.append(pwhtstim[whtstimfrange])

    for findex, fAM in enumerate(fAMs):
        #print(findex)
        #create stimulus and calculate power at fAM for rectified stimulus
        SAM = contrast*np.sin(2*np.pi*fAM*t)
        SAMstimulus = np.sin(2*np.pi*EODf*t) * (1 + SAM)
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        
        #power spectra of the stimulus
        fstim, pstim = welch(np.abs(SAMstimulus-np.mean(SAMstimulus)), fs=1/dt, nperseg=npersegfAM)
        pSAM_interpolator = interpolate(fstim, pstim)
        pfAMs[findex] = pSAM_interpolator(fAM)
        
        #power spectra of the AM
        ffAM, pAM = welch(SAM-np.mean(SAM), fs=1/dt, nperseg=npersegfAM)
        pAM_interpolator = interpolate(ffAM, pAM)
        pfAMAM[findex] = pAM_interpolator(fAM)
    
    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
    fig.suptitle('SAM powers for contrast %f' %(contrast))
    axs[0].set_title('Stimulus')
    axs[1].set_title('Amplitude modulation')
    axs[0].plot(fAMs, pfAMs)
    axs[1].plot(fAMs, pfAMAM)
    
    fig, axw = plt.subplots(1,2, sharex=True, sharey=True)
    fig.suptitle('RAM powers for contrast %f' %(contrast))
    axw[0].set_title('Stimulus')
    axw[1].set_title('Amplitude modulation')
    axw[0].plot(fwhtstim[whtstimfrange], pwhtstim[whtstimfrange])
    axw[1].plot(fwht[whtnoisefrange], pwht[whtnoisefrange])

    print(pwht[whtnoisefrange]/pwhtstim[whtstimfrange])
    print(pfAMAM/pfAMs)
    
    noisetoAMinterpolator = interpolate(fwht, pwht)
    interpwht = noisetoAMinterpolator(fAMs)
    stimtoSAMinterpolator = interpolate(fwhtstim, pwhtstim)
    interpwhtstim = stimtoSAMinterpolator(fAMs)

    print(np.mean(np.array(interpwht/pfAMAM)[:-1]))
    correctionfactor = np.mean(np.array(interpwhtstim/pfAMs)[:-1])
    
    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break
        
        
#Check the power of both stimuli with the given correction factor in the above loop.
for contrast in contrasts:
    whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
    fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
    whtnoisefrange = (fwht>cflow) & (fwht<cfup) #frequency range to plot the power for white nose
    whtnoiseAMpwr.append(pwht[whtnoisefrange])
    
    tRAM = t[1:]
    whtstimulus = np.sin(2*np.pi*EODf*tRAM) * (1 + whtnoise)
    fwhtstim, pwhtstim = welch(np.abs(whtstimulus-np.mean(whtstimulus)), fs=1/dt, nperseg=nperseg)
    whtstimfrange = (fwhtstim>cflow) & (fwhtstim<cfup) #frequency range to plot the power for white nose
    whtnoisespwr.append(pwhtstim[whtstimfrange])
    for findex, fAM in enumerate(fAMs):
        correctionfactor = 0.1220904473654484 / np.sqrt(2.473) 
        """
        The second sqrt term in the correction factor is for setting the RAM stim power equal to SAM stim power. First 
        term only sets wht noise power equal to stim power. Main idea is P_SAMorRAM/P_abs = 2.473 (P_SAMorRAM is the SAM
        or RAM amplitude modulation term power, P_abs is the stimulus power with absolute value taken)
        P_SAMorRAM/P_abs = 2.473 <=> c_abs = c_SAMorRAM / sqrt(2.473) (as power-contrast relation is quadratic)
        
        First correction term cs = cn*a sets P_abss = P_RAM; cs is SAM contrast, cn is RAM contrast
        P_abss = P_RAM <=> P_SAM / 2.473 = P_RAM
        <=> (cn*a)^2 / (2.473*alpha*beta) = cn^2 / (alpha*f_c)
        <=> a^2 = 2.473 * beta / (f_c) <=> a = sqrt(2.473 * beta / (f_c))
        
        In order to get P_abss = P_absn (s stimulus, n noise respectively)
        P_abss = P_absn <=> c_abss^2 / (alpha*beta) = c_absn^2 / (alpha*f_c) 
        *As c_abs = c_SAMorRAM / sqrt(2.473) holds for both SAM and RAM:
        <=> cs^2 / (alpha*beta) = cn^2 / (alpha*f_c)
        <=> cn = cs * sqrt(beta/fc) <=> cn = cs * b
        
        The relationship between a and b is as follows:
        b = a / sqrt(2.473)
        
        Therefore, this correction factor sets both SAM and RAM stimuli power equal.
        """
        #print(findex)
        #create stimulus and calculate power at fAM for rectified stimulus
        SAM = correctionfactor * contrast * np.sin(2*np.pi*fAM*t)
        SAMstimulus = np.sin(2*np.pi*EODf*t) * (1 + SAM)
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        
        #power spectra of the stimulus
        fstim, pstim = welch(np.abs(SAMstimulus-np.mean(SAMstimulus)), fs=1/dt, nperseg=npersegfAM)
        pSAM_interpolator = interpolate(fstim, pstim)
        pfAMs[findex] = pSAM_interpolator(fAM)
    fig, ax = plt.subplots(1,1)
    ax.plot(fwhtstim[whtnoisefrange], pwhtstim[whtnoisefrange], 'k-', label='RAM stimulus')
    ax.plot(fAMs, pfAMs, 'r-', label='SAM stimulus')
    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break
     
"""[0.00306205 0.00572459 0.00608836 0.00620109 0.00633805 0.00633314
 0.00622729 0.00609101 0.00591097 0.00627901 0.0060646  0.00635232
 0.00612766 0.00633339 0.00621184 0.00574212 0.00613691 0.00613817
 0.00604134 0.00586497 0.00295277]
[0.00306204 0.00572473 0.00608829 0.00620096 0.00633818 0.00633366
 0.00622732 0.00609094 0.00591075 0.00627927 0.00604857 0.00635249
 0.00612725 0.00633342 0.00621225 0.00574204 0.00613686 0.00613818
 0.00604134 0.00586496 0.00985831]"""