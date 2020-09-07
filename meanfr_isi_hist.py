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

random.seed(666)

#Stimulus and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings

ISIlist = []
meanspkfr = np.zeros(len(parameters))
tlength = 10 #stop time
tstart = 0.15 #the time point to start plotting (in seconds)
tstop = 0.2 #the time point to stop plotting (in seconds)

for i in range(0, len(parameters)):
    example_cell_idx = i
    print("Example with cell: {}".format(parameters[example_cell_idx]['cell']))
    model_params = parameters[example_cell_idx]
    cell = model_params.pop('cell')
    EODf = model_params.pop('EODf')
    
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = model_params["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    stimulus = np.sin(2*np.pi*frequency*t)

    #Get the stimulus response of the model
    spiketimes = mod.simulate(stimulus, **model_params)
    spikeISI = np.diff(spiketimes)
    ISIlist.append(spikeISI)
    meanspkfr[i] = len(spiketimes) / tlength #mean spike firing rate
    
    print('mean spike firing rate (averaged over stimulus time) is %.3f'%(meanspkfr[i]))
    fig, (axh, axt) = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.suptitle('cell %s' %(cell))
    axh.hist(spikeISI*EODf, bins=np.arange(0,20.2,0.2))
    axh.set_title('ISI histogram')
    axh.set_xlabel('ISI [EOD period]')
    axh.set_ylabel('# of occurence')
    axh.set_xticks(np.arange(0,21,2))
    axh.text(0.65,0.5, 'mean fr: %.2f Hz'%(meanspkfr[i]), size=10, transform=axh.transAxes)
    axt.plot(t[(t>tstart) & (t<tstop)]*1000, stimulus[(t>tstart) & (t<tstop)], '-k', label='stimulus', linewidth=0.8)
    axt.plot(spiketimes[(spiketimes>tstart) & (spiketimes<tstop)]*1000, 
                        np.zeros(len(spiketimes[(spiketimes>tstart) & (spiketimes<tstop)])), '.r', 
                        label='spikes', markersize=2)
    axt.set_title('Stimulus and spikes')
    axt.set_xlabel('Time [ms]')
    axt.set_ylabel('Stimulus amplitude [a.u.]')
    axt.legend(loc='lower right')
    plt.pause(0.5)
    plt.show()
    checker = 0
    while checker == 0:
        a = input('press enter to continue, write esc to quit \n')
        if a == '':
            checker = 1
            plt.close()
        elif a == 'esc':
            checker = 1
        else: 
            a = input('Wrong button, please press enter for the next plot or write esc to terminate. \n')
    if a == 'esc':
        break        

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
        (BLACKBOARD HAS A SKETCH OF HOW STUFF SHOULD LOOK LIKE, HOPE THAT THANG STAYS THERE XD)
    
        +Play around with the model, see how the parameters change/unfold over time, check behavior in different 
        parameters. See also how different voltages behave over time.
        
        +Feel free to do other works you have :) Jan Benda comes at around 14.30 on 08.09
      
"""