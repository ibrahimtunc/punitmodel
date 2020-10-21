# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:02:40 2020

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

#Compare power spectra of SAM and RAM stimuli.

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


for cidx, contrast in enumerate(contrasts):
    print(cidx)
    #create white noise for different contrasts
    whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
    whtnoises[:,cidx] = whtnoise
    #calculate white noise power for different contrasts
    fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
    whtnoisespwr.append(pwht)
    
    #same thing as RAM for the SAM at different contrasts
    #calculate for the given contrast each fAM stimulus and corresponding power
    pfAMs = np.zeros(len(fAMs))
    for findex, fAM in enumerate(fAMs):
        #print(findex)
        stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*fAM*t))
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        fSAM, pSAM = welch(np.abs(stimulus-np.mean(stimulus)), fs=1/dt, nperseg=npersegfAM)
        pSAM_interpolator = interpolate(fSAM, pSAM)
        pfAMs[findex] = pSAM_interpolator(fAM)
    SAMstimpwr.append(pfAMs)

whtnoisespwr = np.array(whtnoisespwr)
SAMstimpwr = np.array(SAMstimpwr)
fig, axps = plt.subplots(3,4, sharex=True, sharey=True)
fig.suptitle('SAM and RAM powers at different contrasts')
lastax = axps[-1,-1]

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


axps = np.delete(axps.reshape(12), 11)
whtnoisefrange = (fwht>cflow) & (fwht<cfup) #frequency range to plot the power for white nose
for idx, ax in enumerate(axps):
    ax.plot(fwht[whtnoisefrange], whtnoisespwr[idx, :][whtnoisefrange], 'k--', label='RAM')
    ax.plot(fAMs, SAMstimpwr[idx,:], 'r--', label='SAM')
    ax.set_title('contrast=%.2f' %(contrasts[idx]))
    lastax.plot(fwht[whtnoisefrange], whtnoisespwr[idx, :][whtnoisefrange])
    lastax.set_ylim([0,0.0012])
axps[4].set_ylabel('Power')
fig.text(0.45, 0.05, 'Frequency [Hz]')
axps[-1].legend(loc='best')
lastax.set_title('RAM all contrasts')
lastaxyticks = np.linspace(0,0.0012,7)
lastax.set_yticks(lastaxyticks)
plt.subplots_adjust(wspace=0.25)
#about to do something super lame:
for idx,tick in enumerate(lastaxyticks):
    fig.text(0.7, 0.105+0.0365*idx, np.round(tick,4))
 
#check the contrast and stimulus power relationship
# Pr = cr^2/(alpha*fc) <=> cr^2/(Pr*fc) = alpha
RAMalpha = np.tile(contrasts,(491,1))**2 / (whtnoisespwr[:,whtnoisefrange].T * (cfup-cflow))
RAMalpha = np.mean(RAMalpha, 0) #take the mean value along the frequencies, as the frequency power fluctuates around 
                                #zero mean. array contains alpha for each contrast
#Ps = cs^2/(alpha*beta) <=> cs^2/(Ps*alpha) = beta , alpha is the same as RAMalpha
SAMbeta = np.mean(np.tile(contrasts,(20,1))**2 / SAMstimpwr[:,1:].T, 0) / RAMalpha 
SAMbeta = np.mean(SAMbeta)
#discard the first SAM frequency, calculate cs^2/Ps, take the mean along the frequencies and finally divide by alpha to 
#get the beta value for each contrast, then take the mean of that as well.

#Now theoretically, cs = cr*sqrt(beta/fc), so for any RAM contrast, multiplying it with sqrt(beta/fc) shall give same 
#stimulus power for SAM.
contrastsRAM = np.linspace(0, 0.5,11)
contrastsSAM = contrastsRAM * np.mean(np.sqrt(SAMbeta/(cfup-cflow)))
whtnoises = np.zeros([len(t)-1,len(contrasts)])
whtnoisespwr = []
SAMstimpwr = []
nperseg = 2**15

for i in range(len(contrastsRAM)):
    print(i)
    #create white noise for different contrasts
    whtnoise = contrastsRAM[i] * helpers.whitenoise(**whitenoiseparams)
    whtnoises[:,cidx] = whtnoise
    #calculate white noise power for different contrasts
    fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
    whtnoisespwr.append(pwht)
    
    #same thing as RAM for the SAM at different contrasts
    #calculate for the given contrast each fAM stimulus and corresponding power
    pfAMs = np.zeros(len(fAMs))
    for findex, fAM in enumerate(fAMs):
        #print(findex)
        stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrastsSAM[i]*np.sin(2*np.pi*fAM*t))
        npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
        fSAM, pSAM = welch(np.abs(stimulus-np.mean(stimulus)), fs=1/dt, nperseg=npersegfAM)
        pSAM_interpolator = interpolate(fSAM, pSAM)
        pfAMs[findex] = pSAM_interpolator(fAM)
    SAMstimpwr.append(pfAMs)

whtnoisespwr = np.array(whtnoisespwr)
SAMstimpwr = np.array(SAMstimpwr)

fig, axps = plt.subplots(3,4, sharex=True, sharey=True)
fig.suptitle('SAM and RAM powers at different contrasts, SAM contrast adjusted')


axps = np.delete(axps.reshape(12), 11)
whtnoisefrange = (fwht>cflow) & (fwht<cfup) #frequency range to plot the power for white nose
for idx, ax in enumerate(axps):
    ax.plot(fwht[whtnoisefrange], whtnoisespwr[idx, :][whtnoisefrange], 'k--', label='RAM')
    ax.plot(fAMs, SAMstimpwr[idx,:], 'r--', label='SAM')
    ax.set_title('RAM contrast=%.2f' %(contrasts[idx]))

axps[4].set_ylabel('Power')
fig.text(0.45, 0.05, 'Frequency [Hz]')
axps[-1].legend(loc='best')
lastax.set_title('RAM all contrasts')
plt.subplots_adjust(wspace=0.25)

