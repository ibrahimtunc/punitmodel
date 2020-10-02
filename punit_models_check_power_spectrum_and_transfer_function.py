# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:50:17 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
from scipy.signal import welch

#Create the peristimulus time histogram for a sinus modulated sinus curve.
random.seed(666)
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

cell_idx = 0
fAMlowpower = [45.55, 60.4, 65.35, 70.3, 85.15, 90.1, 95.05] #AM frequencies showing drop in power in transfer function
#!fAMlowpower list is for cell_idx=0, if you want to check for other cells, you need to change the values (and also
#slightly the code)
decibeltransform = True

#stimulus parameters
tlength = 100
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
contrast = 0.1
contrastfs = np.linspace(1,100,21) #frequency of the amplitude modulation in Hz
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

stimfig, stimaxes =  plt.subplots(5,4, sharex = True, sharey = True)
respfig, respaxes =  plt.subplots(5,4, sharex = True, sharey = True)
stimaxes = np.reshape(stimaxes,20)
respaxes = np.reshape(respaxes,20)
stimfig.suptitle('Stimulus power spectrum, cell %s' %(cell))
respfig.suptitle('Response power spectrum, cell %s' %(cell))
#calculate stimulus
for i, f_contrast in enumerate(contrastfs[1:]):
    print(f_contrast)
    stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*f_contrast*t))
    spiketimes = mod.simulate(stimulus, **cellparams)
    f, p, meanspkfr = helpers.power_spectrum(stimulus, spiketimes, t, kernel, nperseg)
    fstim, pstim = welch(stimulus, nperseg=nperseg, fs=1/t_delta)
    
    if decibeltransform == True:
        p = helpers.decibel_transformer(p)
        pstim = helpers.decibel_transformer(pstim)
    
    pfAM = p[(f>f_contrast-0.3) & (f<f_contrast+0.3)][0]
    psfAM = pstim[(f>EODf-0.3) & (f<EODf+0.3)][0]
    psfAMflank1 = pstim[(f>EODf-f_contrast-0.4) & (f<EODf-f_contrast+0.4)][0]
    psfAMflank2 = pstim[(f>EODf+f_contrast-0.4) & (f<EODf+f_contrast+0.4)][0]
    if np.round(f_contrast, 2) in fAMlowpower:
        respaxes[i].set_facecolor('silver')
        stimaxes[i].set_facecolor('silver')
    respaxes[i].plot(f[(f<150)], p[(f<150)])
    respaxes[i].plot(f_contrast,pfAM,'k.')
    stimaxes[i].plot(fstim[(fstim<EODf+150) & (fstim>EODf-150)], 
                   pstim[(fstim<EODf+150) & (fstim>EODf-150)])
    stimaxes[i].plot(np.array([-f_contrast,0,f_contrast])+EODf,[psfAMflank1,psfAM,psfAMflank2],'r.')
    respaxes[i].set_title('$f_{AM}=%.2f$' %(f_contrast))
    stimaxes[i].set_title('$f_{AM}=%.2f$' %(f_contrast))
    """
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
    """    
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

"""
THIS SCRIPT IS NOW OBSOLETE, ALL TRANSFERRED TO punit_model_power_spectrum.py

"""