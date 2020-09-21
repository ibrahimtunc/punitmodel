# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:51:23 2020

@author: Ibrahim Alperen Tunc
"""

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
import numpy as np
import random
import helper_functions as helpers

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
I_offs = [0, 10, 20]
stimA = 5 #stimulus amplitude, play around
stimAs = [2, 10, 20]
freq = 10 #Frequency of the stimulus, use amplitude modulation frequency 10 Hz
t_delta = 0.00005
t_deltas = [10**-3, 10**-4, 5*10**-6]
t = np.arange(0, tlength, t_delta)

#kernel parameters
kernelparams = {'sigma' : 0.001, 'lenfactor' : 8, 'resolution' : t_delta}
#create kernel
kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)

#check the decay for different tau and refractory values:
taureflist = np.logspace(np.log10(1), np.log10(1000), 20)/1000 #the logarithmic tau and refractory period values

for ioff in I_offs:
    stimulus = stimA * np.sin(2*np.pi*freq*t) + ioff
    decaydf = helpers.tau_ref_scan(taureflist, t, ntrials, example_params, stimulus, kernel)
    dataname = savepath+'\decay_index_tau_refractory_f=%.1f_I_off=%f_tdelta=%f_stimamp=%f_scan_intervals_%f_%f_log.csv'%(freq,
                                                                                           ioff, t_delta, 
                                                                                           stimA, taureflist[0], 
                                                                                           taureflist[-1])
    decaydf.to_csv(dataname, index=False)

for stima in stimAs:
    stimulus = stima * np.sin(2*np.pi*freq*t) + I_off
    decaydf = helpers.tau_ref_scan(taureflist, t, ntrials, example_params, stimulus, kernel)    
    dataname = savepath+'\decay_index_tau_refractory_f=%.1f_I_off=%f_tdelta=%f_stimamp=%f_scan_intervals_%f_%f_log.csv'%(freq,
                                                                                           I_off, t_delta, 
                                                                                           stima, taureflist[0], 
                                                                                           taureflist[-1])
    decaydf.to_csv(dataname, index=False)

for tdelta in t_deltas:
    t = np.arange(0, tlength, tdelta)    
    stimulus = stimA * np.sin(2*np.pi*freq*t) + I_off
    example_params['deltat'] = tdelta
    decaydf = helpers.tau_ref_scan(taureflist, t, ntrials, example_params, stimulus, kernel)    
    dataname = savepath+'\decay_index_tau_refractory_f=%.1f_I_off=%f_tdelta=%f_stimamp=%f_scan_intervals_%f_%f_log.csv'%(freq,
                                                                                           I_off, tdelta, 
                                                                                           stimA, taureflist[0], 
                                                                                           taureflist[-1])
    
    decaydf.to_csv(dataname, index=False)
    
    
t = np.arange(0, tlength, t_deltas[0])    
stimulus = stimA * np.sin(2*np.pi*freq*t) + I_off