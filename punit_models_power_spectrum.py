# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:24:28 2020

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

datafiles = os.listdir('.\data')
idxx = [datafiles[a][0]=='2' for a in range(len(datafiles))]
datafiles = np.array(datafiles)[idxx]
for cell_idx in range(len(parameters)):
    #stimulus parameters
    tlength = 100
    tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
    contrast = 0.1
    contrastf = 50 #frequency of the amplitude modulation in Hz
    
    #model parameters
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
    
    #rest of stimulus parameters depending on model parameters
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    
    #calculate stimulus
    stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*contrastf*t))
    
    #kernel parameters
    kernelparams = {'sigma' : 0.0001, 'lenfactor' : 5, 'resolution' : t_delta}#kernel is muhc shorter for power spectrum
    
    #create kernel
    kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
    
    #power spectrum parameters:
    nperseg = 2**15
    
    spiketimes = mod.simulate(stimulus, **cellparams)
    
    f, p, meanspkfr = helpers.power_spectrum(stimulus, spiketimes, t, kernel, nperseg)
    
    pdB = helpers.decibel_transformer(p)
    
    #stimulus power spectrum
    fstim, pstim = welch(stimulus-np.mean(stimulus), nperseg=nperseg, fs=1/t_delta)#zero peak of power spectrum is part of
                                                                                   #the stimulus, which stays even when 
                                                                                   #stimulus mean is substracted.
    pdBstim = helpers.decibel_transformer(pstim)
    
    #cell f-I curve
    dataframe = pd.read_csv(savepath+'\\'+datafiles[cell_idx])
    vals = dataframe.to_numpy()
    baselinefs = vals[:,0][~np.isnan(vals[:,0])]
    initialfs = vals[:,1][~np.isnan(vals[:,1])]
    steadyfs = vals[:,2][~np.isnan(vals[:,2])]
    contrasts = vals[:,3][~np.isnan(vals[:,3])]
    
    #check for different AM frequencies
    fAMs = np.linspace(0,100,21)
    fAMs[0]+=1
    pfAMs = np.zeros(len(fAMs)) #the power at AM frequencies preallocated.
    for i, fAM in enumerate(fAMs):
        stimuluss = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*fAM*t))
        spiketimes = mod.simulate(stimuluss, **cellparams)
        f, p, __ = helpers.power_spectrum(stimuluss, spiketimes, t, kernel, nperseg)
        pfAMs[i] = p[(f>fAM-0.5) & (f<fAM+0.5)][0]
        
    fig, (axps, axp, axam, axfi) = plt.subplots(1,4)
    fig.suptitle(cell)
    axps.plot(fstim[fstim<1000], pdBstim[f<1000])
    axps.set_xlabel('Frequency [Hz]')
    axps.set_ylabel('Power [dB]')
    axps.set_title('Power spectrum of the stimulus')
    
    axp.plot(f[f<1000], pdB[f<1000])
    axp.plot(f[(f>EODf-0.3) & (f<EODf+0.3)], pdB[(f>EODf-0.3) & (f<EODf+0.3)], '.', label='EODf')
    axp.plot(f[(f>contrastf-0.3) & (f<contrastf+0.3)], pdB[(f>contrastf-0.3) & (f<contrastf+0.3)], '.', label='contrastf')
    axp.plot(f[(f>meanspkfr-0.3) & (f<meanspkfr+0.3)], pdB[(f>meanspkfr-0.3) & (f<meanspkfr+0.3)], '.', label='meanspkfr')
    axp.set_xlabel('Frequency [Hz]')
    axp.set_ylabel('Power [dB]')
    axp.legend()
    axp.set_title('Power spectrum of the response')
    
    axam.plot(fAMs,np.sqrt(pfAMs)/contrast, '.--')#to get as transfer function
    axam.set_xlabel('AM Frequency [Hz]')
    axam.set_ylabel('Power')
    axam.set_title('Transfer function')
    
    helpers.plot_contrasts_and_fire_rates(axfi,contrasts,baselinefs,initialfs,steadyfs)

    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """
    asd = input('press enter to continue ') #way faster than waitforbuttonpress!!!! downside is running from shell
    while asd != '':
        asd = input('Wrong button, press enter please ')
    plt.close()
    """
#add stimulus and f-I curves as plots, check if f_AM peak is wide or narrow in the power spectrum