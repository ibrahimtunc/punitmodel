# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:16:37 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import helper_functions as helpers

#Create the peristimulus time histogram for a sinus curve. Check then if the firing rate decay happens when EOD period is 
#shorter than refractory period

random.seed(666)

parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

#stimulus parameters
ntrials = 100 #number of trials to average over
tlength = 10
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)


for j in range(len(parameters)):
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(j, parameters)
    
    #rest of stimulus parameters depending on model parameters
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    
    #calculate stimulus
    stimulus = np.sin(2*np.pi*frequency*t)
    
    #kernel parameters
    kernelparams = {'sigma' : 0.001, 'lenfactor' : 8, 'resolution' : t_delta}

    #create kernel
    kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
    
    convolvedspklist = np.zeros([t.shape[0],ntrials]) #initialized list of convolved spikes
    spiketrains = np.zeros([t.shape[0],ntrials]) #initialized list of spike trains

    for i in range(ntrials):
        #run the model for the given stimulus and get spike times
        spiketimes, spikeISI, meanspkfr = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
        #spike train with logical 1s and 0s
        spikearray = np.zeros(len(t)) 
        #convert spike times to spike trains
        spikearray[np.digitize(spiketimes,t)-1] = 1 #np.digitize(a,b) returns the index values for a where a==b. For convolution
        #for np.digitize() see https://numpy.org/doc/stable/reference/generated/numpy.digitize.html is this ok to use here?
        #np.digitize returns the index as if starting from 1 for the last index, so -1 in the end
        #convolve the spike train with the gaussian kernel    
        convolvedspikes = np.convolve(kernel, spikearray, mode='same')
        convolvedspklist[:,i] = convolvedspikes
        spiketrains[:,i] = spikearray
        
    peristimulustimehist = np.mean(convolvedspklist, axis=1)
    fig, (axp, axr, axs) = plt.subplots(3,1, sharex = True, figsize = (12, 7))
    
    axp.plot(t[t>0.1]*1000, peristimulustimehist[t>0.1])
    axp.set_title('Peristimulus time histogram')
    axp.set_ylabel('Spiking frequency [Hz]')
    
    axr.plot(t[t>0.1]*1000, spiketrains[t>0.1]*np.arange(1,spiketrains.shape[1]+1).T, 'k.', markersize=1)
    axr.set_ylim(0.8 , ntrials+1)
    axr.set_title('Spike raster')
    axr.set_ylabel('Trial #')
    
    axs.plot(t[t>0.1]*1000, stimulus[t>0.1])
    axs.set_title('Stimulus')
    axs.set_xlabel('time [ms]')
    axs.set_ylabel('Stimulus amplitude [a.u.]')
    print(j, cell)
    if 1/frequency < cellparams['ref_period']:
        print('CELL REFRACTORY PERIOD LONGER THAN STIMULUS PERIOD!!!!')
        print('Noise strength=%f, refractory period=%f, EODf=%d' %(cellparams['noise_strength'],
                                                                   cellparams['ref_period'], frequency))
    plt.pause(0.5)
    plt.show()
    asd = input('press enter to continue ') #way faster than waitforbuttonpress!!!! downside is running from shell
    while asd != '':
        asd = input('Wrong button, press enter please ')
    plt.close()

        
""" done so far: 51 now doing: 52 ERROR AT 52, keep doing from here later on.

The cell idx 10 does not have shorter EOD period than its refractory period, but still there is the decay.
Also other examples where there is no decay but EOD period is shorter than refractory period
cell idx 5 : refractory period longer, still no long time decay in firing rate.
cell idx 7 : refractory period longer, still no long time decay in firing rate.
cell idx 8 : refractory period longer, and long time decay in firing rate (until around 2s).
cell idx 10: refractory period shorter, but long time decay in firing rate (until around 3s).
cell idx 13: refractory period shorter, but long time decay in firing rate (until around 1s).
cell idx 15: refractory period shorter, but long time decay in firing rate (until around 0.5s).
cell idx 16: refractory period longer, and long time decay in firing rate (until around 2s).
cell idx 17: refractory period shorter, but long time decay in firing rate (until around 0.5s).
cell idx 24: refractory period shorter, but long time decay in firing rate (until around 0.5s).
cell idx 26: refractory period shorter, but long time decay in firing rate (until around 2s).
cell idx 29: refractory period shorter, but long time decay in firing rate (until around 4s). 
cell idx 30: refractory period longer, still no long time decay in firing rate.
cell idx 31: refractory period shorter, but long time decay in firing rate (until around 0.5s).
cell idx 34: refractory period shorter, but long time decay in firing rate (until around 1s).
cell idx 38: refractory period shorter, but long time decay in firing rate (until around 1s).
cell idx 45: refractory period shorter, but long time decay in firing rate (until around 1s). very subtle
cell idx 50: refractory period shorter, but long time decay in firing rate (until around 0.5s).
cell idx xx:
cell idx xx:
cell idx xx:
cell idx xx:
cell idx xx:
cell idx xx:
cell idx xx:
cell idx xx:
cell idx xx:
"""