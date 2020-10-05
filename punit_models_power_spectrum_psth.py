# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:12:03 2020

@author: Ibrahim Alperen Tunc
"""


import model as mod
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
from scipy.signal import welch
from scipy.interpolate import interp1d as interpolate
import os 
import pandas as pd

#Create the peristimulus time histogram for a sinus modulated sinus curve.
random.seed(666)
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

datafiles = os.listdir('.\data')
idxx = [datafiles[a][0]=='2' for a in range(len(datafiles))]
datafiles = np.array(datafiles)[idxx]
decibeltransform = False


for cell_idx in range(len(parameters)):
    #stimulus parameters
    tlength = 100
    tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
    contrast = 0.1
    contrastf = 50 #frequency of the amplitude modulation in Hz
    
    #model parameters
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
    print(cell_idx)
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
    power_interpolator_decibel = interpolate(f, pdB)

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
    fAMs = np.linspace(0,300,31)
    plotcutoff = np.max(fAMs)+50 #frequency cutoff for plotting
    fAMs[0]+=1
    pfAMs = np.zeros(len(fAMs)) #the power at AM frequencies preallocated.
    prespfAMs = np.zeros(len(fAMs)) #the power at AM frequencies preallocated, decibel transformed for the plot.
    #power spectra figures for stimulus and response
    stimfig, stimaxes =  plt.subplots(5,6, sharex = True, sharey = True)
    respfig, respaxes =  plt.subplots(5,6, sharex = True, sharey = True)
    stimaxes = np.reshape(stimaxes,30)
    respaxes = np.reshape(respaxes,30)
    stimfig.suptitle('Stimulus power spectrum')
    respfig.suptitle('Response power spectrum, cell %s' %(cell))
        
    for i, fAM in enumerate(fAMs):
        stimuluss = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*fAM*t))
        convolvedspikeslist = []
        for l in range(0,10):
            spiketimess = mod.simulate(stimuluss, **cellparams)
            convolvedspikes, __ = helpers.convolved_spikes(spiketimess, stimulus, t, kernel)
    
            convolvedspikeslist.append(convolvedspikes)
        avgspikes = np.mean(convolvedspikeslist, axis=0)
    
        f, p = welch(avgspikes[t>0.1], nperseg=nperseg, fs=1/t_delta)

        fstim, pstim = welch(stimuluss, nperseg=nperseg, fs=1/t_delta)
        presp = p
        
        if decibeltransform == True:
            presp = helpers.decibel_transformer(p)
            pstim = helpers.decibel_transformer(pstim)
        presp_interpolator = interpolate(f, presp)
        pstim_interpolator = interpolate(f, pstim)
            
        power_interpolator = interpolate(f, p)
        pfAMs[i] = np.max(power_interpolator(np.linspace(fAM-1,fAM+1,101)))
        prespfAMs[i] = np.max(presp_interpolator(np.linspace(fAM-1,fAM+1,101)))

        psfAM = pstim_interpolator(EODf)
        psfAMflank1 = pstim_interpolator(EODf-fAM) 
        psfAMflank2 = pstim_interpolator(EODf+fAM)
        
        if i>0:
            respaxes[i-1].plot(f[(f<plotcutoff)], presp[(f<plotcutoff)])
            respaxes[i-1].plot(fAM,prespfAMs[i],'k.')
            stimaxes[i-1].plot(fstim[(fstim<EODf+plotcutoff) & (fstim>EODf-plotcutoff)], 
                    pstim[(fstim<EODf+plotcutoff) & (fstim>EODf-plotcutoff)])
            stimaxes[i-1].plot(np.array([-fAM, 0, fAM])+EODf,[psfAMflank1,psfAM,psfAMflank2],'r.')
            respaxes[i-1].set_title('$f_{AM}=%.2f$' %(fAM))
            stimaxes[i-1].set_title('$f_{AM}=%.2f$' %(fAM))
     
    #naming and adjusting the stimulus and response plots for each fAM
    if decibeltransform==True:
        stimaxes[8].set_ylabel('Power [db]')
        respaxes[8].set_ylabel('Power [db]')

    else:
        stimaxes[8].set_ylabel('Power')
        respaxes[8].set_ylabel('Power')
    stimaxes[17].set_xlabel('Frequency [Hz]')
    respaxes[17].set_xlabel('Frequency [Hz]')

    stimfig.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.93, wspace=0.1, hspace=0.26)
    respfig.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.92, wspace=0.11, hspace=0.32)
    
    #set the subplot background color to grey where the transfer function shows drop in power    
    [axis.set_facecolor('silver') for axis in respaxes[np.diff(pfAMs)<0]]#frequencies where the power decreases
    [axis.set_facecolor('silver') for axis in stimaxes[np.diff(pfAMs)<0]]#frequencies where the power decreases
    
    #figure for the cell with stimulus/response power spectra, transfer function and f/I curve
    fig, (axps, axp, axam, axfi) = plt.subplots(1,4)
    fig.suptitle(cell)
    fig.text(0.1, 1, 'Power spectra')
    axps.plot(fstim[fstim<1000], pdBstim[f<1000])
    axps.set_xlabel('Frequency [Hz]')
    axps.set_ylabel('Power [dB]')
    axps.set_title('Stimulus ($f_{AM}$=%.2f)'%(contrastf))
    
    axp.plot(f[f<1000], pdB[f<1000])
    axp.plot(EODf, power_interpolator_decibel(EODf), '.', label='EODf')
    axp.plot(contrastf, power_interpolator_decibel(contrastf), '.', label='contrastf')
    axp.plot(meanspkfr, power_interpolator_decibel(meanspkfr), '.', label='meanspkfr')
    axp.set_xlabel('Frequency [Hz]')
    axp.set_ylabel('Power [dB]')
    axp.legend()
    axp.set_title('Response ($f_{AM}$=%.2f)'%(contrastf))
    
    axam.plot(fAMs,np.sqrt(pfAMs)/contrast, '.--')#to get as transfer function
    axam.set_xlabel('AM Frequency [Hz]')
    axam.set_ylabel('Power')
    axam.set_title('Transfer function')
    
    helpers.plot_contrasts_and_fire_rates(axfi,contrasts,baselinefs,initialfs,steadyfs)
    fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.85, wspace=0.25, hspace=0)
    fig.text(0.22,0.9,'Power spectra', fontsize=15)    
    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break

