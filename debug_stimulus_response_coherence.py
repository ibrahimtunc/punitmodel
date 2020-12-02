# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:49:19 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helpers
from scipy.signal import welch, csd, coherence
from scipy.interpolate import interp1d as interpolate

#Debug the average stimulus-response coherence

#General parameters for both stimuli types
#cell and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
contrast = 0.2
tlength = 100
RAMiter = 10 #number of model simulation iterations with RAM stimulus

cflow = 0
cfup = 300
cell_idx=0

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
t = np.arange(0, tlength, dt)
nperseg = 2**12

#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt}#kernel is muhc shorter for power spectrum

#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)

#RAM stimulus for the model
tRAM = t[1:]
whtstimulus = np.sin(2*np.pi*frequency*tRAM) * (1 + whtnoise)

#model response to RAM stimulus   
whtspiketimes = mod.simulate(whtstimulus, **cellparams)
whtspiketimeslist = []
whtspiketimeslist.append(whtspiketimes)
for RAMidx in range(RAMiter-1):
    whtspiketimeslist.append(mod.simulate(whtstimulus, **cellparams))  #for response-response coherence

convolvedspikes = []   
prs = [] #response powers
csdrs = [] #array of all coherence values to be taken mean
welchpairs = [] #array containing the power spectra response products for welch pairs


fs, ps = welch(whtnoise, nperseg=nperseg, fs=1/dt)
for i in range(len(whtspiketimeslist)):
    convolvedspike, __ = helpers.convolved_spikes(whtspiketimeslist[i], whtstimulus, tRAM, kernel)
    fr, pr = welch(convolvedspike, nperseg=nperseg, fs=1/dt)
    if i == 0:
        finterval = (fr>cflow) & (fr<cfup)
    convolvedspikes.append(convolvedspike[tRAM>0.1])
    pr = np.array(pr)
    prs.append(pr[finterval])
    fcoh, psr = csd(convolvedspike[tRAM>0.1], whtnoise[tRAM>0.1], nperseg=nperseg, fs=1/dt)
    csdr = np.array(psr)[finterval]
    welchpair = pr[finterval] * ps[finterval]
    csdrs.append(csdr)
    welchpairs.append(welchpair)
    
convolvedspikes = np.array(convolvedspikes)
prs = np.array(prs)

gammars = np.abs(np.mean(csdrs, 0))**2 / np.mean(welchpairs, 0)

plt.plot(np.sqrt(gammars))
