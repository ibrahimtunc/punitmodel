# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:39:28 2020

@author: Ibrahim Alperen Tunc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:53:56 2020

@author: Ibrahim Alperen Tunc
"""
import os
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
#Reduce the integrate & fire neuron to 1D (no adaptation)

random.seed(666)
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'


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
def simulate(stimulus, deltat=0.00005, v_zero=0.0, threshold=1.0, v_base=0.0,
             mem_tau=0.015, noise_strength=0.05, ref_period=0.001):
    """ Simulate a P-unit (1D reduced integrate and fire neuron).

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
    v_mems = np.zeros(len(stimulus))
    for i in range(len(stimulus)):
        v_mem += (v_base - v_mem + stimulus[i]
                  + noise[i]) / mem_tau * deltat #membrane voltage (integrate & fire) v_base additive there to bring zero
                                                 #voltage value of v_mem to baseline                                                
        # refractory period:
        if len(spike_times) > 0 and (deltat * i) - spike_times[-1] < ref_period + deltat/2:
            v_mem = v_base #v_base is the resting membrane potential.

        # threshold crossing:
        if v_mem > threshold:
            v_mem = v_base
            spike_times.append(i * deltat)
        v_mems[i] = v_mem
    return v_mems, np.array(spike_times)


#load the cells
parameters = load_models('models.csv')
cell_idx = 10
cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)

#example parameters (noiseD is to be played around)
noiseD = 0.1
example_params = { 'v_zero' : 0,
                 'threshold' : 1,
                 'mem_tau' : 0.05,
                 'noise_strength' : noiseD*10,
                 'deltat' : 0.00005, 
                 'v_base' : 0.0,
                 'ref_period' : 0.5}

#stimulus parameters
ntrials = 100 #number of trials to average over
tlength = 10
I_off = 2 #Offset of the stimulus current. Play around with it
stimA = 5 #stimulus amplitude, play around
freqs = np.logspace(np.log10(1), np.log10(1000), 4) #Frequency of the stimulus, use amplitude modulation frequency (10 Hz to start with)
t_delta = cellparams['deltat']

t = np.arange(0, tlength, t_delta)

#try with example parameters:
#v_mems, spiketimes = simulate(stimulus, **example_params)

#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 8, 'resolution' : t_delta}
#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

#check the decay for different tau and refractory values:
taureflist = np.logspace(np.log10(1), np.log10(1000), 20)/1000 #the logarithmic tau and refractory period values


for freq in freqs:
    samefreq = False

    for dataname in os.listdir(savepath)[:-1]:

        while dataname[:5] == 'decay':
            try: #check if the frequency value is in the data name
                dataname.index(str(freq))
            except ValueError:
                break #if freq not in data name get out of the while loop
            if dataname.index(str(freq)) == 29: #check if the frequency is in the right location of the dartaname
                samefreq = True
                print('Scan for this frequency (%.1f) is already done' %(freq))
                break
            else:
                break

    if samefreq == True:
        continue
    
    print('Frequency is %.1f Hz' %(freq))
    period = 1/freq #period length of the stimulus
    stimulus = stimA * np.sin(2*np.pi*freq*t) + I_off
    
    decayIndex = np.zeros([len(taureflist),len(taureflist)]) #columns for tau, rows for refractory

    for idxt, tau in enumerate(taureflist): #tau
        for idxr, ref in enumerate(taureflist): #refractory
    
            convolvedspklist = np.zeros([t.shape[0],ntrials]) #initialized list of convolved spikes
            spiketrains = np.zeros([t.shape[0],ntrials]) #initialized list of spike trains
    
            example_params['mem_tau'] = tau
            example_params['ref_period'] = ref
            print('tau=%f ref=%f' %(tau, ref))
    
            for i in range(ntrials):
                example_params['v_zero'] = np.random.rand()
                v_mems, spiketimes = simulate(stimulus, **example_params)
                
                convolvedspikes, spikearray = helpers.convolved_spikes(spiketimes, stimulus, t, kernel)
                
                convolvedspklist[:,i] = convolvedspikes
                spiketrains[:,i] = spikearray
            
            peristimulustimehist = np.mean(convolvedspklist, axis=1)
            decayidx = np.max(peristimulustimehist[(t<1) & (t>0.15)]) / np.max(peristimulustimehist[(t>=9) & (t<9.84995)])
            decayIndex[idxt, idxr] = decayidx
            
    
    decaydf = pd.DataFrame(decayIndex)
    dataname = savepath+'\decay_index_tau_refractory_f=%.1f_scan_intervals_%f_%f_log.csv'%(freq,
                                                                                           taureflist[0], taureflist[-1]) 
    decaydf.to_csv(dataname, index=False) #rows are refractory period, columns are membrane tau

#check for speed
#import timeit
#create some dummy list for checking speed

#setup = """
#import numpy as np
#t_delta = 5e-05
#dummyt = np.arange(0,2,t_delta)
#dummyspkt = np.squeeze(sorted(np.random.rand(500,1)))*2 #random numbers between 0-2000 to simulate 500 spikes
#dummyspkarray = np.zeros(len(dummyt))""" 


#start trying different approaches with timeit
#tdig = timeit.timeit('dummyspkarray[np.digitize(dummyspkt,dummyt)-1]', setup = setup, number=10000) 
#tidx = timeit.timeit('dummyspkarray[(dummyspkt//t_delta).astype(np.int)]', setup = setup, number=10000) 
#thist = timeit.timeit('np.histogram(dummyspkt, dummyt)', setup = setup, number=10000) 
#tidx < tdig < thist (0.2, 0.8, 10)

"""
pcolormesh alternative to imshow!
#https://www.kite.com/python/examples/1870/matplotlib-change-x-axis-tick-labels
#check out the link above for modifying ticks
fig, ax = plt.subplots(1,1)
img = ax.imshow(decayIndex)#, extent = [np.min(np.log10(taureflist)),np.max(np.log10(taureflist)),
                           #           np.max(np.log10(taureflist)),np.min(np.log10(taureflist))])

ax.set_xticks(np.round(np.log10(taureflist[0::10]),2))
ax.set_yticks(np.round(np.flip(np.log10(taureflist[0::10])),2))
ax.xaxis.tick_top()
fig.colorbar(img, ax=ax)
#plt.gca().invert_yaxis()
ax.set_ylabel('membrane tau [$\log_{10}$]')
ax.set_xlabel('refractory period [$\log_{10}$]')
"""
