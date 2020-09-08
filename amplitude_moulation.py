# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:33:45 2020

@author: Ibrahim Alperen Tunc
"""
import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers

random.seed(666)


#Amplitude modulation of the stimulus and how the cell behaves to that.
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
tlength = 1.5 #stimulus time length (in seconds)
boxonset = 0.5 #the time point to start AM (in seconds)
cell_idx = 10 #index of the cell of interest.
contrast = -0.5 #the amplitude modulation index (soon to be -.5 to .5 with .1 stepping)
ntrials = 50 #number of trials to average over
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)


cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
t_delta = cellparams["deltat"] #time step in seconds
t = np.arange(0, tlength, t_delta)
#Create a box function which onsets at t=3 and stays like that till the end.
boxfunc = np.zeros(len(t))
boxfunc[(t>=boxonset)] = 1
stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*boxfunc)
"""
#check if cell looks nice
helpers.stimulus_ISI_plotter(cell,t, EODf, stimulus, spiketimes, spikeISI, meanspkfr)
"""
freqs = np.zeros([t.shape[0], ntrials])
for i in range(ntrials):
    spiketimes, spikeISI, meanspkfr = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
    freq = helpers.calculate_isi_frequency(spiketimes,t)
    freqs[:,i] = freq

meanfreq, baselinef, initialidx, initialf, steadyf = \
                helpers.calculate_AM_modulated_firing_rate_values(freqs, t, tstart, boxonset)
"""    
meanfreq = np.mean(freqs,1) #mean firing rate over time as inverse of ISI

baselinef = np.mean(meanfreq[(t>=tstart) & (t<=boxonset)]) #average baseline frequency before amplitude modulation
initialidx = np.squeeze(np.where(abs(meanfreq-baselinef)==np.max(abs(meanfreq-baselinef))))[0] #index of the initial freq
initialf = meanfreq[initialidx] #the initial frequency after amplitude modulation
steadyf = np.mean(meanfreq[(t>=boxonset+0.1)]) #steady state average firing rate
"""


fig, (axf, axs, axq) = plt.subplots(1, 3)

axf.plot(t[t>=tstart]*1000, meanfreq[t>=tstart])
axf.plot(t[initialidx]*1000, initialf, '.r')
axf.plot(t[(t>=tstart) & (t<=boxonset)]*1000, np.ones(len(t[(t>=tstart) & (t<=boxonset)]))*baselinef, 'k-')
axf.plot(t[t>=boxonset+0.5]*1000, np.ones(len(t[t>=boxonset+0.5]))*steadyf, 'k-')
axf.set_title('Mean frequency (n=%d) over time' %(ntrials))
axf.set_xlabel('Time [ms]')
axf.set_ylabel('Mean frequency over trials [1/ISI]')
axs.plot(t[t>=tstart]*1000, stimulus[t>=tstart], label='stimulus', linewidth=0.1)
axs.plot(spiketimes[spiketimes>=tstart]*1000, np.zeros(len(spiketimes[spiketimes>=tstart])), '.', label='spikes')
axs.set_title('Stimulus and spikes')
axs.set_xlabel('Time [ms]')
axs.set_ylabel('Stimulus amplitude')
axs.legend(loc='lower right')

#now do the above over multiple contrasts
contrasts = np.linspace(-0.5,0.5,40)

baselinefs = np.zeros(contrasts.shape)
initialfs = np.zeros(contrasts.shape)
steadyfs = np.zeros(contrasts.shape)
for k, contrast in enumerate(contrasts):
    stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*boxfunc)
    freqs = np.zeros([t.shape[0], ntrials])
    for i in range(ntrials):
        spiketimes, spikeISI, meanspkfr = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
        freq = helpers.calculate_isi_frequency(spiketimes,t)
        freqs[:,i] = freq    
    meanfreq, baselinef, initialidx, initialf, steadyf = \
            helpers.calculate_AM_modulated_firing_rate_values(freqs, t, tstart, boxonset)

    baselinefs[k] = baselinef
    initialfs[k] = initialf
    steadyfs[k] = steadyf
    
axq.plot(contrasts, baselinefs, label='baseline fr')
axq.plot(contrasts, initialfs, label='initial peak fr')
axq.plot(contrasts, steadyfs, label='steady state fr')
axq.set_title('Effect of contrasts')
axq.set_xlabel('Contrast')
axq.set_ylabel('Firing rate [1/ISI]')
axq.legend(loc='lower right')
