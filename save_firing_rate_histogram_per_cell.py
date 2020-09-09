# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:25:12 2020

@author: Ibrahim Alperen Tunc
"""
import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
import pandas as pd
import os

savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'

random.seed(666)
#Run through each cell and save the firing rate and histogram values for each cell for later use
cell_idx=0 #for now
#Load model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

#Amplitude modulation parameters. For explanation of each see helpers.amplitude_modulation
ampmodinputs = {'tlength' : 1.5,
                'boxonset' : 0.5,   
                'contrasts' : np.linspace(-0.5,0.5,40),
                'ntrials' : 100, 
                'tstart' : 0.1}
tlength = 10 #stimulus time length (in seconds, for histogram)

for i, __ in enumerate(parameters):
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(i, parameters)
    dataname = savepath+'\%s_firing_rates_and_ISI_hist.csv' %(cell)
    #Skip if file exists and is readable 
    #https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
    if os.path.isfile(dataname) and os.access(dataname, os.R_OK):
        print("File exists and is readable")
        continue
        
    #Amplitude modulation
    baselinefs, initialfs, steadyfs = helpers.amplitude_modulation(cellparams, EODf, **ampmodinputs)
    
    #Histogram
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    stimulus = np.sin(2*np.pi*frequency*t)#simple sine wave for the hist
    
    __, spikeISI, __ = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
    spikeISI = spikeISI*EODf
    ISIhist, ISIbins = np.histogram(spikeISI, bins=np.arange(0,20.2,0.2))
    
    datarate = {'baselinefs' : baselinefs, 'initialfs' : initialfs, 'steadyfs' : steadyfs, 
                'contrasts' : ampmodinputs['contrasts']}
    
    datahist = {'ISIhist' : ISIhist, 'ISIbins' : ISIbins[1:]}
    dataISI = {'spikeISI' : spikeISI}
    prdf = pd.DataFrame(datarate)
    phistdf = pd.DataFrame(datahist)
    pISIdf = pd.DataFrame(dataISI)    
    dataframe = pd.concat([prdf, phistdf, pISIdf], axis=1)
    dataframe.to_csv(dataname, index=False)


'''
plt.figure()
plt.hist(spikeISI, bins=np.arange(0,20.2,0.2))
plt.figure()
plt.plot(ampmodinputs['contrasts'], initialfs)
plt.plot(ampmodinputs['contrasts'], baselinefs)
plt.plot(ampmodinputs['contrasts'], steadyfs)
'''

"""
np.save(datapath, array) another option to use 
"""

#pdf.to_csv(savepath+'\%s_hist_and_firing_rates.csv' %(cell), index=False)
