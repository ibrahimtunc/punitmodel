# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:34:58 2020

@author: Ibrahim Alperen Tunc
Plot the mean fire rates and ISIs for all neurons
"""
runpath = r'D:\ALPEREN\Tübingen NB\Semester 3\Benda\git\punitmodel'

import sys
sys.path.insert(0, runpath)#!Change the directory accordingly
import model as mod
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as helpers

random.seed(666)

#Stimulus and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

ISIlist = []
meanspkfrs = []
tlength = 10 #stimulus time length (in seconds)
tstart = 0.15 #the time point to start plotting (in seconds)
tstop = 0.2 #the time point to stop plotting (in seconds)
cell_idx = 0 #index of the cell of interest.

#Run from here on, if you break and want to re-see what is up.
checkertotal = 1 #this to keep looking further to cells after breaking.
while checkertotal == 1:
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
    
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    stimulus = np.sin(2*np.pi*frequency*t)

    spikevals = helpers.stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)#spikevals is spiketimes, 
                                                                                        #spikeISI, meanspkfr
    ISIlist.append(spikevals[1])
    meanspkfrs.append(spikevals[2]) #mean spike firing rate per second

    helpers.stimulus_ISI_plotter(cell, t, EODf, stimulus, *spikevals, tstart=tstart, tstop=tstop)
    
    checkerplot = 0
    while checkerplot == 0:
        a = input('press enter to continue, write esc to quit \n')
        if a == '':
            checkerplot = 1
            plt.close()
        elif a == 'esc':
            checkerplot = 1
        else: 
            a = input('Wrong button, please press enter for the next plot or write esc to terminate. \n')
    if a == 'esc' or cell_idx == len(parameters)-1:
        checkertotal = 0
        cell_idx += 1
        break        
    cell_idx += 1




"""
TODO:   +add mean fire rate in hist DONE
      
        +Normalize ISI axis in form of EODf DONE
      
        +Plot the sinus and spikes also in a second axis DONE
      
        +Make a model where probability to get a spike in a cycle is fixed number (0.2). For that take random number 
        generator (randn) and do logical indexing np.where(). Poisson spike train DONE in poisson_spike.py
      
        +Do amplitude modulation on the sinus with different contrasts (contrast is the same as amplitude modulation).
        The idea is using a step function -> sin()*(1+step) where the amp of the step function varies. The code main.py
        has a function which gives you the firing rate as a function of time, use that to get the firing rate. For a 
        given cell (choose yourself a nice one), do multiple trials of amplitude modulated stimulus and average over 
        multiple trials (15-20 trials). Then on this average find the baseline value (where there is no amp modulation,
        average firing rate there, actually what you had in the loop of this script) and get the highest deviation from
        that (positive or negative). This will be the initial firing rate. Then get the steady state firing rate, where
        the firing rate stays +- constant although stimulus is still amplitude modulated (the offset response is btw not 
        of relevance). Then plot those values against the contrast (amplitude modulation) values to see what's up.
        (BLACKBOARD HAS A SKETCH OF HOW STUFF SHOULD LOOK LIKE, HOPE THAT THANG STAYS THERE XD) DONE
    
        +Play around with the model, see how the parameters change/unfold over time, check behavior in different 
        parameters. See also how different voltages behave over time.
        
        +Do the steady state initial and baseline firing rate to all cells, and store the dataset for each cell in 
        separate panda dataframe csv file somehow. Also save the histogram values for each cell (use np.histogram to
        get the values and plot them separately, hist is in this script). Then in a separate script plot for each cell 
        the histogram and the firing rate curves (2 subplots so far), this to show the heterogeneity of the population.
        DONE
        
        +Do the amplitude modulation now with a sinusoidal, stimulus is EODf and contrast is sinusoidal with frequency
        50 Hz (a*sin(2pi*f*t)) (then increase up to 400 Hz) and keep the stimulus short (300 ms sufficient) and do 
        lots and lots of trials. Then, get spikes with the integrate and fire model for a given cell, convolve those 
        spikes with a Gaussian (if you are causality freak use Poisson or Gamma where its zero for negative values). 
        Use Gaussian kernel, where the mean t value is substituted with t_i (spike time). Start with gaussian sigma=1 ms
        and kernel width 8*sigma, and play around with the values of gaussian sigma, kernel length and time resolution 
        (time resolution can be 0.1 ms). Finally, after understanding the convolution (do also your cricket homework of 
        how the stimulus and filter are placed relative to each other the phase thingy by plotting the spikes and 
        filtered response), do this for 100 trials and calculate the mean spiking probability over time for the given
        stimulus. This is peristimulus time histogram (PSTH). Unconvolved spike train is a list of logical, the convolved
        one is list of floats, the over each trial get the average value for the given time window and this averaged time
        array is the PSTH, plot together with SAM stimulus.
"""