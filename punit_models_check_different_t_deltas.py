# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:10:18 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import helper_functions as helpers
import pandas as pd

#Create the peristimulus time histogram for a sinus curve. Check then punit model behavior for different delta t values

random.seed(666)
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

#stimulus parameters
ntrials = 100 #number of trials to average over
tlength = 10
tstart = 0.1 #get rid of the datapoints from 0 until this time stamp (in seconds)
frequency = 10 #Stimulus frequency in Hz, keep it low against Nyquist aliasing effects.
t_deltas = [5*10**-4, 10**-4, 5*10**-6]

decayidxs = np.zeros([len(parameters), len(t_deltas)])

for i in range(len(parameters)):
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(i, parameters)
    frequency = EODf
    for j, tdelta in enumerate(t_deltas):
        t = np.arange(0, tlength, tdelta)
        cellparams["deltat"] = tdelta
        #calculate stimulus
        stimulus = np.sin(2*np.pi*frequency*t)
        
        #kernel parameters
        kernelparams = {'sigma' : 0.001, 'lenfactor' : 8, 'resolution' : tdelta}
    
        #create kernel
        kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
        
        
        convolvedspklist = np.zeros([t.shape[0],ntrials]) #initialized list of convolved spikes
        spiketrains = np.zeros([t.shape[0],ntrials]) #initialized list of spike trains

        for k in range(ntrials):
            #run the model for the given stimulus and get spike times
            spiketimes, spikeISI, meanspkfr = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
            #spike train with logical 1s and 0s
            spikearray = np.zeros(len(t)) 
            #convert spike times to spike trains
            spikearray[(spiketimes//(t[1]-t[0])).astype(np.int)] = 1
            convolvedspikes = np.convolve(kernel, spikearray, mode='same')
            convolvedspklist[:,k] = convolvedspikes
            spiketrains[:,k] = spikearray
            
        peristimulustimehist = np.mean(convolvedspklist, axis=1)
        
        decayidx = np.max(peristimulustimehist[(t<1) & (t>0.15)]) / np.max(peristimulustimehist[(t>=9) & (t<9.84995)])
        decayidxs[i,j] = decayidx

decaydf = pd.DataFrame(decayidxs, columns = t_deltas)
dataname = savepath+'\punit_cells_decayindex_deltat=%s.csv'%(t_deltas)
decaydf.to_csv(dataname)

longdecaycellidxs = [8, 10, 13, 15, 16, 17, 24, 26, 29, 31, 34, 38, 45, 59, 61, 66] #indexes of cells showing long decay
thresholdedlongdecaycellidxs = [16, 26, 29, 34, 59]

fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
fig.suptitle('Effect of integration step on long time decay')
for i, ax in enumerate(axs):
    ax.plot(decaydf[t_deltas[i]],range(0,len(decaydf)), 'k.')
    ax.plot(decaydf[t_deltas[i]][longdecaycellidxs], longdecaycellidxs, 'r.')
    ax.plot(decaydf[t_deltas[i]][thresholdedlongdecaycellidxs], thresholdedlongdecaycellidxs, 'b.')
    ax.set_title('$\delta t= %f$'%(t_deltas[i]))
axs[0].set_xlabel('Long time decay')
axs[0].set_ylabel('Cell index')
nodecay = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                          markersize=10, label='No decay')
longdecay = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
                          markersize=10, label='Long decay (manual)')

longdecayauto = mlines.Line2D([], [], color='blue', marker='.', linestyle='None',
                          markersize=10, label='Long decay (auto)')

axs[-1].legend(handles=[nodecay,longdecay, longdecayauto], loc='upper right')
