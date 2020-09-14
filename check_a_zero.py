# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:12:08 2020

@author: Ibrahim Alperen Tunc
"""

#Check the a_zero values for the given cell parameters.

import numpy as np
import matplotlib.pyplot as plt
import copy
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
             delta_a=0.08, tau_a=0.1, v_offset=-10.0, mem_tau=0.015, noise_strength=0.05,
             input_scaling=60.0, dend_tau=0.001, ref_period=0.001):
    """ Simulate a P-unit.

    Returns
    -------
    adapt: float
        a_zero adaptation variable value in the last instance.
    adapts: 1-D array
        a_zero adaptation variable values over the entire time.
    spike_times: 1-D array
        Simulated spike times in seconds.
    """ 
    #print(deltat,v_zero, a_zero, threshold, v_base, delta_a, tau_a, v_offset, mem_tau, noise_strength, input_scaling
    #      , dend_tau, ref_period, EODf, cell)
    
    # initial conditions:
    v_dend = stimulus[0]
    v_mem = v_zero
    adapt = a_zero

    # prepare noise:    
    noise = np.random.randn(len(stimulus))
    noise *= noise_strength / np.sqrt(deltat) # scale white noise with square root of time step, coz else they are 
                                              # dependent, this makes it time step invariant.
    # rectify stimulus array:
    stimulus = stimulus.copy()
    stimulus[stimulus < 0.0] = 0.0

    # integrate:
    spike_times = []
    adapts = np.zeros(len(stimulus))
    for i in range(len(stimulus)):
        v_dend += (-v_dend + stimulus[i]) / dend_tau * deltat #dendrite voltage, think as input
        v_mem += (v_base - v_mem + v_offset + 
                    v_dend * input_scaling - adapt + noise[i]) / mem_tau * deltat #membrane voltage (integrate & fire)
        adapt += -adapt / tau_a * deltat #adaptation component

        # refractory period:
        if len(spike_times) > 0 and (deltat * i) - spike_times[-1] < ref_period + deltat/2:
            v_mem = v_base

        # threshold crossing:
        if v_mem > threshold:
            v_mem = v_base
            spike_times.append(i * deltat)
            adapt += delta_a / tau_a
        adapts[i] = adapt

    return adapt, adapts, np.array(spike_times)


parameters = load_models('models.csv')
tlength = 100 #length of t - keep it long as adaptation somehow takes too long, you will accordingly adjust a_zero
newparameters = copy.deepcopy(parameters)
for i, __ in enumerate(parameters):
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(i, parameters)
    cellparams['v_zero'] = np.random.rand()
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    stimulus = np.sin(2*np.pi*frequency*t)
    
    a_zero, a_zeros, spiketimes = simulate(stimulus, **cellparams)
    newparameters[i]['v_zero'] = cellparams['v_zero']
    newparameters[i]['cell'] = cell
    newparameters[i]['EODf'] = EODf
    newparameters[i]['a_zero'] = a_zero
    fig, axa = plt.subplots(1,1)
    axa.plot(t, a_zeros)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break
#a_zero is exactly the same! so this value is correct
azero_diffs = np.zeros(len(parameters))
for i in range(len(parameters)):
    azero_diff = parameters[i]['a_zero'] - newparameters[i]['a_zero']
    azero_diffs[i] = azero_diff
    
rms_azero = np.sqrt(np.mean(azero_diffs**2))
plt.plot(azero_diffs, '.')

newparamsdf = pd.DataFrame(newparameters)
newparamsdf.to_csv('newmodel.csv', index=False)
