# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:59:01 2020

@author: Ibrahim Alperen Tunc
"""

#Helper functions

import model as mod
import numpy as np
import matplotlib.pyplot as plt


def parameters_dictionary_reformatting(cell_idx, parameters):
    """
    Reformat the parameter dictionary so that it can be nicely passed to other functions.
    
    Parameters
    ----------
    cell_idx: integer
        Index of the cell of interest
    parameters: dictionary
        The dictionary of parameters for all cells
    
    Returns
    -------
    cell: string
        The name of the cell
    Eodf: float
        The EOD frequency of the cell model
    cellparams: dictionary
        The dictionary of parameters for the chosen cell
    """
    print("Example with cell: {}".format(parameters[cell_idx]['cell']))
    cellparams = parameters[cell_idx]
    cell = cellparams.pop('cell')
    EODf = cellparams.pop('EODf')
    return cell, EODf, cellparams

    
def stimulus_ISI_calculator(cellparams, stimulus, tlength=10):
    """
    Calculate ISI for a given cell and stimulus
    
    Parameters
    ----------
    cellparams: dictionary
        The model parameters of the cells
    stimulus: 1-D array
        The stimulus of interest.
    tlength: Float
        Length of the stimulus time in seconds
        
    Returns
    -------
    spiketimes: 1D array
        The timestamp of the spike times
    spikeISI: 1D array
        The ISI values for the spikes
    meanspkfr: float
        The average spiking rate #TODO: add the average fire rate as a function of time (inverse of ISI)
    """
    
    #Get the stimulus response of the model
    spiketimes = mod.simulate(stimulus, **cellparams)
    spikeISI = np.diff(spiketimes)
    meanspkfr = len(spiketimes) / tlength #mean spike firing rate per second
    
    print('mean spike firing rate (averaged over stimulus time) is %.3f'%(meanspkfr))
    return spiketimes, spikeISI, meanspkfr
    

def stimulus_ISI_plotter(cell, t, EODf, stimulus, spiketimes, spikeISI, meanspkfr, tstart=0.15, tstop=0.2):
    """
    Plot the ISI and stimulus together with the spikes
    
    Parameters
    ----------
    cell: string
        The name of the cell at hand
    t: 1D array
        The time array for the plot
    EODf: float
        The EOD frequency of the cell model
    stimulus: 1-D array
        The stimulus of interest.
    spiketimes: 1D array
        The timestamp of the spike times
    spikeISI: 1D array
        The ISI values for the spikes
    meanspkfr: float
        The average spiking rate
    tstart: float
        Onset of the stimulus in seconds which is to be plotted
    tstop: float
        Offset of the stimulus in seconds which is to be plotted
    """
    fig, (axh, axt) = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.suptitle('cell %s' %(cell))
    axh.hist(spikeISI*EODf, bins=np.arange(0,20.2,0.2))
    axh.set_title('ISI histogram')
    axh.set_xlabel('ISI [EOD period]')
    axh.set_ylabel('# of occurence')
    axh.set_xticks(np.arange(0,21,2))
    axh.text(0.65,0.5, 'mean fr: %.2f Hz'%(meanspkfr), size=10, transform=axh.transAxes)
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
    return


def calculate_isi_frequency(spiketimes, t):
    """
    Calculate ISI frequency
    Do the following: either calculate the instantaneous fire rate over multiple trials, or use ISI 
    For ISI: the spike frequency is the same between each spike timepoints. For smoothing averaging will be done
    For now ISI is used, maybe other one can also be implemented soon.
    
    Parameters
    -----------
    spikes: 1D array
        Spike time points in seconds
    t: 1D array
        The time stamps in seconds
        Returns
    ----------
    freq: 1D array
        Frequency trace which starts at the time of first spike and ends at the time of the last spike.
    """
    spikeISI = np.diff(spiketimes)
    freq = 1/spikeISI
    freqtime = np.zeros(len(t))
    freqtime[0:np.squeeze(np.where(t==spiketimes[0]))]=freq[0]#initialize the frequency as 1/isi for the first spike (from onset on)
    for i in range(len(freq)):
        tbegin = int(np.where(t==spiketimes[i])[0])
        try:
            tend = int(np.where(t==spiketimes[i]+spikeISI[i])[0])
        except TypeError:
            freqtime[tbegin:] = freq[i]
            return freqtime
        freqtime[tbegin:tend] = freq[i]
    freqtime[tend:] = freq[i]
    return freqtime


def calculate_AM_modulated_firing_rate_values(freqs, t, tstart, boxonset):
    """
    Calculate the baseline, steady state and initial firing rate for a given frequency trace.
    
    Parameters
    ----------
    freqs: 1D array
        The frequency trace (see calculate_ISI_frequency)
    t: 1D array
        Time stamps in seconds
    tstart: float
        The time onset in seconds, discard the beginning until here
    boxonset: float
        The start of the amplitude modulation
   
    Returns
    -------
    meanfreq: 1D array
        The mean firing rate value over time (as 1/ISI)
    baselinef: float
        The mean firing rate before amplitude modulation
    initialidx: integer
        Index location of initialf
    initialf: float
        The peak firing rate at amplitude modulation
    steadyf: float
        The adapted firing rate during amplitude modulation
    """
    meanfreq = np.mean(freqs,1) #mean firing rate over time as inverse of ISI

    baselinef = np.mean(meanfreq[(t>=tstart) & (t<=boxonset)]) #average baseline frequency before amplitude modulation
    initialidx = np.argmax(np.abs(meanfreq[t>=boxonset]-baselinef)) #index of the initial freq
    initialf = meanfreq[t>=boxonset][initialidx] #the initial frequency after amplitude modulation
    #if np.abs(initialf - baselinef) < np.std(meanfreq[(t>=tstart) & (t<=boxonset)]) * 5:
    #    print("im in" + str(np.std(meanfreq[(t>=tstart) & (t<=boxonset)])))
    #    initialf = baselinef
    steadyf = np.mean(meanfreq[(t>=boxonset+0.5)]) #steady state average firing rate
    return meanfreq, baselinef, initialidx+int(np.where(t==boxonset)[0]), initialf, steadyf