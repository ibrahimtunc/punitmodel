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
        The average spiking rate
    """
    
    #Get the stimulus response of the model
    cellparams['v_zero'] = np.random.rand()
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
    fig, (axh, axt) = plt.subplots(1,2, figsize=(12,5))
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
    if spiketimes[-1] < t[-1]-0.5: #if last spike does not occur in the last 500 ms, set the firing rate to zero.
        freqtime[tend:] = 0
    else:
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


def amplitude_modulation(cellparams, EODf, tlength, boxonset, contrasts, ntrials, tstart):
    """
    Calculate the values of the steady state, initial and baseline firing rates of a given cell for a subset of 
    amplitude modulation
    
    Parameters
    ----------
    cellparams: dictionary
        The parameter dictionary containing values for a single cell except EODf and cell name    
    EODf: float
        The electric organ discharge frequency in Hz
    tlength: float
        Stimulus length in seconds
    boxonset: float
        Onset time of the amplitude modulation in seconds    
    contrasts: array/list
        The list of amplitudes for the amplitude modulation. For positive values the stimulus amplitude increases,  
        for negative it decreases and for zero it stays constant.
    ntrials: integer
        Number of trials to calculate the frequencies
    tstart: float
        The time onset, before which the data is discarded. This is to give the model sufficient time to accomodate.
           
    Returns
    --------
    baselinefs: 1D array
        The average baseline firing rate before amplitude modulation
    initialfs: 1D array
        The maximum firing rate with the amplitude modulation
    steadyfs: 1D array
        The average firing rate after adaptation to amplitude modulation
    """
    frequency = EODf #Electric organ discharge frequency in Hz, used for stimulus
    t_delta = cellparams["deltat"] #time step in seconds
    t = np.arange(0, tlength, t_delta)
    #Create a box function which onsets at t=3 and stays like that till the end.
    boxfunc = np.zeros(len(t))
    boxfunc[(t>=boxonset)] = 1
    baselinefs = np.zeros(contrasts.shape)
    initialfs = np.zeros(contrasts.shape)
    steadyfs = np.zeros(contrasts.shape)
    
    for k, contrast in enumerate(contrasts):
        stimulus = np.sin(2*np.pi*frequency*t) * (1 + contrast*boxfunc)
        freqs = np.zeros([t.shape[0], ntrials])
        for i in range(ntrials):
            spiketimes, spikeISI, meanspkfr = stimulus_ISI_calculator(cellparams, stimulus, tlength=tlength)
            freq = calculate_isi_frequency(spiketimes,t)
            freqs[:,i] = freq    
    
        __, baselinef, __, initialf, steadyf = \
                calculate_AM_modulated_firing_rate_values(freqs, t, tstart, boxonset)

        baselinefs[k] = baselinef
        initialfs[k] = initialf
        steadyfs[k] = steadyf
    
    return baselinefs, initialfs, steadyfs


def plot_contrasts_and_fire_rates(ax, contrasts, baselinefs, initialfs, steadyfs):
    """
    Plot the firing rates for a given amplitude modulation contrast
    
    Parameters
    ----------
    ax: Axes object
        The axis to plot
    contrasts: array/list
        The list of amplitudes for the amplitude modulation. For positive values the stimulus amplitude increases,  
        for negative it decreases and for zero it stays constant.
    baselinefs: 1D array
        The average baseline firing rate before amplitude modulation
    initialfs: 1D array
        The maximum firing rate with the amplitude modulation
    steadyfs: 1D array
        The average firing rate after adaptation to amplitude modulation
        
    Returns
    -------
    """
    ax.plot(contrasts*100, baselinefs, label='$f_b$')
    ax.plot(contrasts*100, initialfs, label='$f_0$')
    ax.plot(contrasts*100, steadyfs, label='$f_{\infty}$')
    ax.set_title('Effect of contrasts')
    ax.set_xlabel('Contrast [%]')
    ax.set_ylabel('Firing rate [1/ISI]')
    ax.set_ylim(-1, 1200)
    ax.legend(loc='upper left')
    return


def plot_ISI(ax, spikeISI, meanspkfr):
    """
    Plot the ISI histogram alone
    
    Parameters
    ----------
    ax: Axes object
        The axis to plot
    spikeISI: 1D array
        The ISI values for the spikes, EODf normalized
    meanspkfr: float
        The average spiking rate
    
    Returns
    --------
    """
    ax.hist(spikeISI, bins=np.arange(0,20.2,0.2))
    ax.set_title('ISI histogram')
    ax.set_xlabel('ISI [EOD period]')
    ax.set_ylabel('# of occurence')
    ax.set_xticks(np.arange(0,21,2))
    ax.text(0.65,0.5, 'mean fr: %.2f Hz'%(meanspkfr), size=10, transform=ax.transAxes)
    return


def spike_gauss_kernel(sigma, lenfactor, resolution):
    """
    The Gaussian kernel for spike convolution
    
    Parameters
    ----------
    sigma: float
        The kernel width in s
    lenfactor: float
        The size of the kernel in terms of sigma
    resolution: float
        The time resolution in s. Keep it same with the stimulus resolution
    
    Returns
    -------
    kernel: 1D array
        The Gaussian convolution kernel
    t: 1D array
        Time window of the kernel
    """
    t = np.arange(-sigma*lenfactor/2, sigma*lenfactor/2+resolution, resolution) 
    #t goes from -sigma/2*lenfactor to +sigma/2*lenfactor, +resolution because arange stops prematurely. 
    #the start and stop point of t is irrelevant, but only kernel is used
    kernel = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-((t)**2) / (2*sigma**2)) 
    #maximum of the Gaussian kernel is in the middle
    return kernel, t


def convolved_spikes(spiketimes, stimulus, t, kernel):
    """
    Convolve the spikes with a given kernel
    
    Parameters
    ----------
    spiketimes: 1-D array
        The array containing spike occurence times (in seconds)
    stimulus: 1-D array
        The stimulus array
    t: 1-D array
        The time array in seconds
    kernel: 1-D array
        The kernel array
        
    Returns
    --------
    convolvedspikes: 1-D array
        The array containing convolved spikes
    spikearray: 1-D array
        The logical array containing 1 for the time point where spike occured.
    """
    #run the model for the given stimulus and get spike times
    #spike train with logical 1s and 0s
    spikearray = np.zeros(len(t)) 
    #convert spike times to spike trains
    spikearray[(spiketimes//(t[1]-t[0])).astype(np.int)] = 1
    
    #spikearray[np.digitize(spiketimes,t)-1] = 1 #np.digitize(a,b) returns the index values for a where a==b. For convolution
    #for np.digitize() see https://numpy.org/doc/stable/reference/generated/numpy.digitize.html is this ok to use here?
    #np.digitize returns the index as if starting from 1 for the last index, so -1 in the end THIS stays just in case FYI
    
    #convolve the spike train with the gaussian kernel    
    convolvedspikes = np.convolve(kernel, spikearray, mode='same')
    return convolvedspikes, spikearray