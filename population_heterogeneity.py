# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:08:42 2020

@author: Ibrahim Alperen Tunc
"""
import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers
import pandas as pd
import os

random.seed(666)
#Plot histograms and firing rates for different cells with subplots next to each other
savepath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel\data'

datafiles = os.listdir('.\data')
idxx = [datafiles[a][0]=='2' for a in range(len(datafiles))]
datafiles = np.array(datafiles)[idxx]
idx = 0 #index of the cell
tlength = 10 #stimulus time length (in seconds, for histogram) 
#tlength is from save_firing_rate_histogram_per_cell.py as of 09.09.2020

checkertotal = 1 #this to keep looking further to cells after breaking.
while checkertotal == 1:
    dataframe = pd.read_csv(savepath+'\\'+datafiles[idx])
    vals = dataframe.to_numpy()
    baselinefs = vals[:,0][~np.isnan(vals[:,0])]
    initialfs = vals[:,1][~np.isnan(vals[:,1])]
    steadyfs = vals[:,2][~np.isnan(vals[:,2])]
    contrasts = vals[:,3][~np.isnan(vals[:,3])] 
    ISIhist = vals[:,4][~np.isnan(vals[:,4])]
    ISIbins = vals[:,5][~np.isnan(vals[:,5])]
    spikeISI = vals[:,6][~np.isnan(vals[:,6])]
    meanspkfr = (len(spikeISI)+1) / tlength
    
    fig, (axh, axq) = plt.subplots(1,2, figsize=(12,5))
    fig.suptitle('cell %s' %(datafiles[idx][:22]))
    helpers.plot_contrasts_and_fire_rates(axq,contrasts,baselinefs,initialfs,steadyfs)
    helpers.plot_ISI(axh,spikeISI, meanspkfr)
    checkerplot = 0
    plt.pause(0.5)
    plt.show()
    while checkerplot == 0:
        a = input('press enter to continue, write esc to quit \n')
        if a == '':
            checkerplot = 1
            plt.close()
        elif a == 'esc':
            checkerplot = 1
        else: 
            a = input('Wrong button, please press enter for the next plot or write esc to terminate. \n')
    if a == 'esc' or idx == len(datafiles)-1:
        checkertotal = 0
        idx += 1
        break        
    idx += 1

"""    
TODO:   +Add mean firing rate, compute it from ISI, inverse by using cumsum (1/mean(ISI), check it out)
        The mean firing rate over time is the same as 1/mean(ISI), as 1/mean(ISI) = n_spikes / sum(ISI) where sum(ISI) ~ t
        For more precision, just take the number of elements in the ISI and add 1 to get to the number of spikes, then 
        divide that with the time window. DONE
        
        +Contrasts as percentages DONE
        
        +Same y axis for all firing rates DONE set to 1500
        
        +Check the cell shooting 4000 Hz, parameters, etc. DONE
        cell is '2012-05-10-ad-invivo-1', EODf 891.62 Hz
        specialities : 'mem_tau' = 0.0007168215913677159 very small time constant
                       'ref_period': 3.413210716933666e-05 very short refractory period
        all parameters as below:
            {'a_zero': 3.482049934191402,
             'delta_a': 0.01647570951108223,
             'dend_tau': 0.0014910069211114215,
             'input_scaling': 24.219477859072562,
             'mem_tau': 0.0007168215913677159,
             'noise_strength': 0.008693794598630195,
             'ref_period': 3.413210716933666e-05,
             'deltat': 5e-05,
             'tau_a': 0.05269970394103512,
             'threshold': 1.0,
             'v_base': 0.0,
             'v_offset': -3.90625,
             'v_zero': 0.0}
        
        +Legend shorten and put top left DONE
"""
