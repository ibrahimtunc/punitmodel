# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:53:56 2020

@author: Ibrahim Alperen Tunc
"""

#Reduce the integrate & fire neuron to 2D

import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
import pandas as pd
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit

random.seed(666)


def load_models(file):
    """ Load model parameter from csv file.

    Parameters
    ----------
    file: string
        Name of file with model parameters.

    Returns
    -------
    parameters: list of dict
        For each cell a dictionary with model parameters.
    """
    parameters = []
    with open(file, 'r') as file:
        header_line = file.readline()
        header_parts = header_line.strip().split(",")
        keys = header_parts
        for line in file:
            line_parts = line.strip().split(",")
            parameter = {}
            for i in range(len(keys)):
                parameter[keys[i]] = float(line_parts[i]) if i > 0 else line_parts[i]
            parameters.append(parameter)
    return parameters


#adjust the simulate function
@jit(nopython=True)
def simulate(stimulus, deltat=0.00005, v_zero=0.0, a_zero=2.0, threshold=1.0, v_base=0.0,
             delta_a=0.08, tau_a=0.1, mem_tau=0.015, noise_strength=0.05,
             ref_period=0.001):
    """ Simulate a P-unit (2D reduced integrate and fire neuron).

    Returns
    -------
    v_mems: 1-D arrray
        Membrane voltage over time.
    adapts: 1-D array
        a_zero adaptation variable values over the entire time.
    spike_times: 1-D array
        Simulated spike times in seconds.
    """ 
    #print(deltat,v_zero, a_zero, threshold, v_base, delta_a, tau_a, v_offset, mem_tau, noise_strength, input_scaling
    #      , dend_tau, ref_period, EODf, cell)
    
    # initial conditions:
    v_mem = v_zero #starting membrane potential
    adapt = a_zero #starting adaptation value

    # prepare noise:    
    noise = np.random.randn(len(stimulus))
    noise *= noise_strength / np.sqrt(deltat) # scale white noise with square root of time step, coz else they are 
                                              # dependent, this makes it time step invariant.
    """
    # rectify stimulus array:
    stimulus = stimulus.copy()
    stimulus[stimulus < 0.0] = 0.0
    """
    # integrate:
    spike_times = []
    adapts = np.zeros(len(stimulus))
    v_mems = np.zeros(len(stimulus))
    for i in range(len(stimulus)):
        v_mem += (v_base - v_mem - adapt + stimulus[i]
                  + noise[i]) / mem_tau * deltat #membrane voltage (integrate & fire) v_base additive there to bring zero
                                                 #voltage value of v_mem to baseline
                                                 
        adapt += -adapt / tau_a * deltat #adaptation component

        # refractory period:
        if len(spike_times) > 0 and (deltat * i) - spike_times[-1] < ref_period + deltat/2:
            v_mem = v_base #v_base is the resting membrane potential.

        # threshold crossing:
        if v_mem > threshold:
            v_mem = v_base
            spike_times.append(i * deltat)
            adapt += delta_a / tau_a
        adapts[i] = adapt
        v_mems[i] = v_mem
    return v_mems, adapts, np.array(spike_times)


#load the cells
parameters = load_models('models.csv')
cell_idx = 10
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
cellparams.pop('dend_tau')
cellparams.pop('input_scaling')
cellparams.pop('v_offset')

#example parameters (noiseD is to be played around)
noiseD = 0.1
example_params = { 'v_zero' : 0,
                 'threshold' : 1,
                 'mem_tau' : 0.005,
                 'tau_a' : 1.00,
                 'noise_strength' : noiseD*10,
                 'deltat' : 0.00005, 
                 'a_zero' : 0.0, 
                 'threshold' : 1.0, 
                 'v_base' : 0.0,
                 'delta_a' : 0.0, 
                 'ref_period' : 0.05}

#stimulus parameters
ntrials = 100 #number of trials to average over
tlength = 10
I_off = 2 #Offset of the stimulus current. Play around with it
stimA = 5 #stimulus amplitude, play around
freq = 30 #Frequency of the stimulus, use amplitude modulation frequency (10 Hz to start with)
period = 1/freq #period length of the stimulus
t_delta = cellparams['deltat']
cellparams['deltat'] = t_delta

t = np.arange(0, tlength, t_delta)
stimulus = stimA * np.sin(2*np.pi*freq*t) + I_off

#try with example parameters:
#v_mems, adapts, spiketimes = simulate(stimulus, **example_params)


#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 8, 'resolution' : t_delta}
#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

convolvedspklist = np.zeros([t.shape[0],ntrials]) #initialized list of convolved spikes
spiketrains = np.zeros([t.shape[0],ntrials]) #initialized list of spike trains

#choose different time points of the stimulus (peak, through and minimun from start and end)
fig2, axts = plt.subplots(3,3, sharex = True, sharey = True)
axts = np.reshape(axts,9)

tvmemlists = [[] for _ in range(len(axts))] #empty list for v_mem for given timepoints of interest
tadaptlists = [[] for _ in range(len(axts))] #empty list for adapts for given timepoints of interest
   
#time indexes for the offset before maxmimum, maximum, offset after maximum and minimum
offbefmaxidx = np.arange(0, tlength/t_delta, period/t_delta).astype(int)
maxtidx = np.arange(period/4/t_delta, tlength/t_delta, period/t_delta).astype(int)
offafmaxidx = np.arange(period/2/t_delta, tlength/t_delta, period/t_delta).astype(int)
mintidx = np.arange(3*period/4/t_delta, tlength/t_delta, period/t_delta).astype(int)
t1idx = int(len(maxtidx) / (freq*2)) #first time index

timeindexes = [] #time indexes of interest for the v_mem - adapts plot
for i in [t1idx*2, int(len(maxtidx)/2), -t1idx*2]:
    timeindexes.append(maxtidx[i])
    timeindexes.append(offafmaxidx[i])  
    timeindexes.append(mintidx[i])

for i in range(ntrials):
    example_params['v_zero'] = np.random.rand()
    v_mems, adapts, spiketimes = simulate(stimulus, **example_params)

    convolvedspikes, spikearray = helpers.convolved_spikes(spiketimes, stimulus, t, kernel)

    convolvedspklist[:,i] = convolvedspikes
    spiketrains[:,i] = spikearray

    for j, tvmemlist in enumerate(tvmemlists):
        tvmemlist.append(v_mems[timeindexes[j]])
        tadaptlists[j].append(adapts[timeindexes[j]])

peristimulustimehist = np.mean(convolvedspklist, axis=1)
fig, (axp, axr, axs) = plt.subplots(3,1, sharex = True)

axp.plot(t[t>0.1]*1000, peristimulustimehist[t>0.1])
axp.set_title('Peristimulus time histogram (reduced model)')
axp.set_ylabel('Spiking frequency [Hz]')

axr.plot(t[t>0.1]*1000, spiketrains[t>0.1]*np.arange(1,spiketrains.shape[1]+1).T, 'k.', markersize=1)
axr.set_ylim(0.8 , ntrials+1)
axr.set_title('Spike raster')
axr.set_ylabel('Trial #')
               
axs.plot(t[t>0.1]*1000, stimulus[t>0.1])
axs.set_title('Stimulus')
axs.set_xlabel('time [ms]')
axs.set_ylabel('Stimulus amplitude [a.u.]')

for k, ax in enumerate(axts):
    ax.plot(tvmemlists[k], tadaptlists[k], 'k.', markersize=5)
    ax.set_title('t=%.4f' %(t[timeindexes[k]]))
axts[3].set_ylabel('Adaptation')
axts[7].set_xlabel('$V_{mem}$')

"""
TODO:   +play around wtih refractory period as well as stimulus frequency and amplitude
!!! Found out that without adaptation, when refractory period is longer than the stimulus frequency, and when there is
enough high noise, there is an exponential decay on the firing rate over a very long time. Read the paper by Knight
Over time, the phase locking to the stimulus gets kind of disturbed, and the variance of the spike timepoint increases
(wider bell shaped curve with maximum located at smaller firing rates)
Decay also happens regardless if the stimulus goes into negative or only stays at positive
"""

"""
STUPID APPROACH STAYS JUST SO THAT YOU KNOW HOW NOT TO CODE 
Stupid approach, use your knowledge on the frequency and sine wave.
timeindexes = [] #time indexes of interest for the v_mem - adapts plot
time1 = int(len(np.where(stimulus==np.max(stimulus))[0])/(freq*5))
for i in [time1, time1*10, -time1]:
    timeindexes.append(np.where(stimulus==np.max(stimulus))[0][i])
    timeindexes.append(np.where((stimulus<=(np.max(stimulus)+np.min(stimulus)+0.001)/2) & 
                                (stimulus>=(np.max(stimulus)+np.min(stimulus)-0.001)/2))\
                                [0][2*i+1]) #as long as i is even this should give the middle value after maximum value  
    timeindexes.append(np.where(stimulus==np.min(stimulus))[0][i])
"""
