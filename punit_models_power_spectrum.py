# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:24:28 2020

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

#Create the peristimulus time histogram for a sinus modulated sinus curve.
random.seed(666)
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

datafiles = os.listdir('.\data')
idxx = [datafiles[a][0]=='2' for a in range(len(datafiles))]
datafiles = np.array(datafiles)[idxx]
decibeltransform = True


#stimulus parameters
tlength = 100
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
contrast = 0.1
contrastf = 50 #frequency of the amplitude modulation in Hz
    
for cell_idx in range(len(parameters)):
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
    kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : t_delta}#kernel is muhc shorter for power spectrum
    
    #create kernel
    kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
    
    #power spectrum parameters:
    nperseg = 2**15
    
    spiketimes = mod.simulate(stimulus, **cellparams)
    
    fexample, pexample, meanspkfr = helpers.power_spectrum(stimulus, spiketimes, t, kernel, nperseg)
    
    pdB = helpers.decibel_transformer(pexample)
    power_interpolator_decibel = interpolate(fexample, pdB)

    #stimulus power spectrum
    fexamplestim, pexamplestim = welch(stimulus-np.mean(stimulus), 
                                       nperseg=nperseg, fs=1/t_delta)#zero peak of power spectrum is part of the stimulus, 
                                                                     #which stays even when stimulus mean is substracted.
                                                                     #take absolute value to get the envelope
    pdBstim = helpers.decibel_transformer(pexamplestim)
    
    #cell f-I curve
    dataframe = pd.read_csv(savepath+'\\'+datafiles[cell_idx])
    vals = dataframe.to_numpy()
    baselinefs = vals[:,0][~np.isnan(vals[:,0])]
    initialfs = vals[:,1][~np.isnan(vals[:,1])]
    steadyfs = vals[:,2][~np.isnan(vals[:,2])]
    contrasts = vals[:,3][~np.isnan(vals[:,3])]
    
    #check for different AM frequencies
    fAMs = np.logspace(np.log10(1),np.log10(300),31)
    plotcutoff = np.max(fAMs)+50 #frequency cutoff for plotting
    fAMs[0]+=1
    pfAMs = np.zeros(len(fAMs)) #the power at AM frequencies preallocated. 
    pStimEODfandAMs = np.zeros([len(fAMs),3]) #power at fAM and EODf of the stimulus envelope, check for variation
    prespfAMs = np.zeros(len(fAMs)) #the power at AM frequencies preallocated, decibel transformed for the plot.
    #power spectra figures for stimulus and response
    stimfig, stimaxes =  plt.subplots(5,6, sharex = True, sharey = True)
    respfig, respaxes =  plt.subplots(5,6, sharex = True, sharey = True)
    stimaxes = np.reshape(stimaxes,30)
    respaxes = np.reshape(respaxes,30)
    stimfig.suptitle('Stimulus power spectrum')
    respfig.suptitle('Response power spectrum, cell %s' %(cell))
        
    for i, fAM in enumerate(fAMs):
        AMwave =  contrast*np.sin(2*np.pi*fAM*t)
        stimuluss = np.sin(2*np.pi*frequency*t) * (1 + AMwave)
        spiketimes = mod.simulate(stimuluss, **cellparams)
        #npersegfAM = 2**15
        #Reduce the stimulus power spectrum fluctuation at different fAMs by adjusting the nperseg. This adjustment
        #ensures to keep nperseg around 2**15 while cut-off windows do not interrupt the stimulus cycle at fAM. This is
        #why there is more fluctuation at fEOD as the nperseg interrupts the cycle at fEOD.
        #(np.max(pStimEODfandAMs[1:,1])-np.min(pStimEODfandAMs[1:,1]))/np.mean(pStimEODfandAMs[1:,1]) this line is
        #is to check the degree of stimulus power fluctuation in percentage (normalized by mean value, first entry is 
        #discarded where fAM is 1 Hz, dont know why but there the power is bit too high compared to rest)
        npersegfAM = np.round(2**(15+np.log2(t_delta*fAM))) * 1/(t_delta*fAM) 
        T = 1 / fAM
        print(npersegfAM / T, np.log2(npersegfAM))
        f, p, __ = helpers.power_spectrum(stimuluss, spiketimes, t, kernel, npersegfAM)
        fstim, pstim = welch(stimuluss-np.mean(stimuluss), nperseg=npersegfAM, fs=1/t_delta)
        fSAM, pSAM = welch(AMwave, nperseg=npersegfAM, fs=1/t_delta)
        presp = p
        pstimm = pstim
        if decibeltransform == True:
            presp = helpers.decibel_transformer(p)
            pstimm = helpers.decibel_transformer(pstim)
        presp_interpolator = interpolate(f, presp)
        pstimm_interpolator = interpolate(f, pstimm)
        pAM_interpolator = interpolate(fSAM, pSAM)
        
        power_interpolator = interpolate(f, p)
        pfAMs[i] = power_interpolator(fAM)
        prespfAMs[i] = presp_interpolator(fAM)

        psfAMEODf = pstimm_interpolator(EODf)
        psfAM = pstimm_interpolator(EODf-fAM) 
        pstim_interpolator = interpolate(f, pstim)
        pStimEODfandAMs[i,0] = pstim_interpolator(EODf)
        pStimEODfandAMs[i,1] = pstim_interpolator(fAM)
        pStimEODfandAMs[i,2] = pAM_interpolator(fAM)
        if i>0:
            respaxes[i-1].plot(f[(f<plotcutoff)], presp[(f<plotcutoff)])
            respaxes[i-1].plot(fAM,prespfAMs[i],'k.')
            stimaxes[i-1].plot(fstim[(fstim<EODf+plotcutoff)], pstimm[(fstim<EODf+plotcutoff)])
            stimaxes[i-1].plot(np.array([EODf-fAM, EODf]), [psfAM,psfAMEODf],'r.')
            respaxes[i-1].set_title('$f_{AM}=%.2f$' %(fAM))
            stimaxes[i-1].set_title('$f_{AM}=%.2f$' %(fAM))
     
    #naming and adjusting the stimulus and response plots for each fAM
    if decibeltransform==True:
        stimaxes[12].set_ylabel('Power [db]')
        respaxes[12].set_ylabel('Power [db]')

    else:
        stimaxes[12].set_ylabel('Power')
        respaxes[12].set_ylabel('Power')
    stimaxes[26].set_xlabel('Frequency [Hz]')
    respaxes[26].set_xlabel('Frequency [Hz]')

    stimfig.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.93, wspace=0.1, hspace=0.26)
    respfig.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.92, wspace=0.11, hspace=0.32)
    
    #set the subplot background color to grey where the transfer function shows drop in power    
    [axis.set_facecolor('silver') for axis in respaxes[np.diff(pfAMs)<0]]#frequencies where the power decreases
    [axis.set_facecolor('silver') for axis in stimaxes[np.diff(pfAMs)<0]]#frequencies where the power decreases
    
    #figure for stimulus power at different EODf+-fAM frequencies.
    fig, axseodf = plt.subplots(1,1)
    #axseodf.plot(fAMs,pStimEODfandAMs[:,0], '.--', label='$2*f_{EOD}$')
    axseodf.plot(fAMs,pStimEODfandAMs[:,1], '.--', label='$f_{AM}$')
    axseodf.set_xlabel('AM Frequency [Hz]')
    axseodf.set_ylabel('Power')
    axseodf.set_title('Stimulus power')
    axseodf.legend()
    
    #figure for the cell with stimulus/response power spectra, transfer function and f/I curve
    fig, (axps, axp, axam, axfi) = plt.subplots(1,4)
    fig.suptitle(cell)
    fig.text(0.1, 1, 'Power spectra')
    axps.plot(fexamplestim[fexamplestim<2000], pdBstim[fexamplestim<2000])
    axps.set_xlabel('Frequency [Hz]')
    axps.set_ylabel('Power [dB]')
    axps.set_title('Stimulus ($f_{AM}$=%.2f)'%(contrastf))
    
    axp.plot(fexample[fexample<1000], pdB[fexample<1000])
    axp.plot(EODf, power_interpolator_decibel(EODf), '.', label='EODf')
    axp.plot(contrastf, power_interpolator_decibel(contrastf), '.', label='contrastf')
    axp.plot(meanspkfr, power_interpolator_decibel(meanspkfr), '.', label='meanspkfr')
    axp.set_xlabel('Frequency [Hz]')
    axp.set_ylabel('Power [dB]')
    axp.legend()
    axp.set_title('Response ($f_{AM}$=%.2f)'%(contrastf))
    
    axam.plot(fAMs,np.sqrt(pfAMs/pStimEODfandAMs[:,2]), '.--') # to get as transfer function (divide by stimulus envelope
                                                               # power at fAM)
                                                               
    axam.set_xlabel('AM Frequency [Hz]')
    axam.set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
    axam.set_title('Transfer function')
    
    helpers.plot_contrasts_and_fire_rates(axfi,contrasts,baselinefs,initialfs,steadyfs)
    fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.85, wspace=0.25, hspace=0)
    fig.text(0.22,0.9,'Power spectra', fontsize=15)    

    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break
    """
    asd = input('press enter to continue ') #way faster than waitforbuttonpress!!!! downside is running from shell
    while asd != '':
        asd = input('Wrong button, press enter please ')
    plt.close()
    """
#add stimulus and f-I curves as plots, check if f_AM peak is wide or narrow in the power spectrum