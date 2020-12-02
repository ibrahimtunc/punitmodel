# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:58:50 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helpers
from scipy.signal import welch
from scipy.interpolate import interp1d as interpolate
import pandas as pd
import matplotlib as mpl
from cycler import cycler

#Script for the presentation figures:

#General figure parameters:
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 20,
           'xtick.labelsize' : 15,
           'ytick.labelsize' : 15,
           'legend.fontsize' : 15}
plt.style.use(figdict)

#cell and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
cell_idx = 0
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
dt = cellparams['deltat']
tlength = 100
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
t = np.arange(0, tlength, dt)
frequency = EODf

#Figure 1: EOD sine wave and spikes
sinewave = np.sin(2*np.pi*frequency*t)
tinterval = (t<0.13) & (t>tstart)
sinespiketimes = mod.simulate(sinewave, **cellparams)
fig1, ax1 = plt.subplots(1,1)
ax1.plot(t[tinterval]*1000, sinewave[tinterval])
ax1.plot((sinespiketimes)*1000, np.zeros(len(sinespiketimes)), 'k|', mew=2, ms=20)
ax1.plot([t[tinterval][0]*1000, t[tinterval][-1]*1000], [0,0], 'k', linewidth = 0.8)
ax1.set_title('Response to sine wave with EOD frequency (%.1f)' %(frequency))
ax1.set_xlabel('Time [ms]')
ax1.set_ylabel('Stimulus amplitude')
ax1.set_xlim([t[tinterval][0]*1000-0.05, t[tinterval][-1]*1000+0.05])
ax1.set_ylim(np.array([-1.05,1.05]))

#Figure 2: 1x3 plot of stimulus, spikes and f-I curve
tlength = 1.5 #stimulus time length (in seconds)
boxonset = 0.5 #the time point to start AM (in seconds)
contrast = 0.3 #the amplitude modulation contrast
ntrials = 100 #number of trials to average over
t = np.arange(0, tlength, dt)


#Create a box function which onsets at t=3 and stays like that till the end.
boxfunc = np.zeros(len(t))
boxfunc[(t>=boxonset)] = 1*contrast
stimulus = np.sin(2*np.pi*frequency*t) * (1 + boxfunc)
tinterval = (t>0.4) & (t<0.6) 

freqs = np.zeros([t.shape[0], ntrials])
for i in range(ntrials):
    boxspiketimes, spikeISI, meanspkfr = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
    freq = helpers.calculate_isi_frequency(boxspiketimes,t)
    freqs[:,i] = freq

meanfreq, baselinef, initialidx, initialf, steadyf = \
                helpers.calculate_AM_modulated_firing_rate_values(freqs, t, tstart, boxonset)

fig2, ax2 = plt.subplots(1, 3)

boxtimes = boxspiketimes[(boxspiketimes<0.6) & (boxspiketimes>0.4)]

ax2[0].plot(t[tinterval]*1000, stimulus[tinterval], 'dimgrey', linewidth=0.1)
ax2[0].plot(t[tinterval]*1000, boxfunc[tinterval]+1, 'purple', label='AM')
ax2[0].plot((boxtimes)*1000, np.zeros(len(boxtimes)), 'k|', mew=1, ms=20, label='spikes')
ax2[0].set_title('Stimulus and spikes')
ax2[0].set_xlabel('Time [ms]')
ax2[0].set_ylabel('Stimulus amplitude')
ax2[0].legend(loc='upper left')

ax2[1].plot(t[t>tstart]*1000, meanfreq[t>tstart], 'k')
ax2[1].plot(t[initialidx]*1000, initialf, '.g')
ax2[1].plot(t[(t>tstart) & (t<=boxonset)]*1000, np.ones(len(t[(t>tstart) & (t<=boxonset)]))*baselinef, 'b-')
ax2[1].plot(t[t>=boxonset+0.5]*1000, np.ones(len(t[t>=boxonset+0.5]))*steadyf, 'r-')
ax2[1].set_title('Mean firing rate (n=%d)' %(ntrials))
ax2[1].set_xlabel('Time [ms]')
ax2[1].set_ylabel('Mean firing rate over trials [1/ISI]')


contrasts = np.linspace(-0.5,0.5,40)
baselinefs, initialfs, steadyfs = helpers.amplitude_modulation(cellparams, EODf, tlength, boxonset, 
                                                                       contrasts, ntrials, tstart)
ax2[2].plot(contrasts, baselinefs, 'b-', label='$f_{b}$')
ax2[2].plot(contrasts, initialfs, 'g-', label='$f_{0}$')
ax2[2].plot(contrasts, steadyfs, 'r-', label='$f_{\infty}$')
ax2[2].set_title('Effect of contrasts')
ax2[2].set_xlabel('Contrast')
ax2[2].set_ylabel('Firing rate [1/ISI]')
ax2[2].legend(loc='lower right')

