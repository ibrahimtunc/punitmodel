# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:50:17 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
from scipy.signal import welch
import os 
import pandas as pd

#Create the peristimulus time histogram for a sinus modulated sinus curve.
random.seed(666)
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

cell_idx = 5

#stimulus parameters
tlength = 100
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
contrast = 0.1
contrastfs = np.linspace(1,100,11) #frequency of the amplitude modulation in Hz
#model parameters
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
#rest of stimulus parameters depending on model parameters
frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
t_delta = cellparams["deltat"] #time step in seconds
t = np.arange(0, tlength, t_delta)

#kernel parameters
kernelparams = {'sigma' : 0.0001, 'lenfactor' : 5, 'resolution' : t_delta}#kernel is muhc shorter for power spectrum
    
#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
    
#power spectrum parameters:
nperseg = 2**15

#calculate stimulus
for i, f_contrast in enumerate(contrastfs):
    print(f_contrast)
    stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*f_contrast*t))
    spiketimes = mod.simulate(stimulus, **cellparams)
    f, p, meanspkfr = helpers.power_spectrum(stimulus, spiketimes, t, kernel, nperseg)
    fstim, pstim = welch(stimulus-np.mean(stimulus), nperseg=nperseg, fs=1/t_delta)
    pfAM = p[(f>f_contrast-0.5) & (f<f_contrast+0.5)][0]
    psfAM = p[(f>EODf-0.5) & (f<EODf+0.5)][0]
    psfAMflank1 = p[(f>EODf-f_contrast-0.5) & (f<EODf-f_contrast+0.5)][0]
    psfAMflank2 = p[(f>EODf+f_contrast-0.5) & (f<EODf+f_contrast+0.5)][0]
    fig, (axs, axr) = plt.subplots(1,2)
    axr.plot(f[f<1000], p[f<1000])
    axs.plot(fstim[fstim<1000], pstim[fstim<1000])
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break