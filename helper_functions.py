# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:59:01 2020

@author: Ibrahim Alperen Tunc
"""

#Helper functions

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch, csd, coherence
from scipy.interpolate import interp1d as interpolate
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit


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
    ax.text(0.65,0.5, 'mean fr: %.2f Hz'%(meanspkfr), size=15, transform=ax.transAxes)
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


@jit(nopython=True)
def simulate_1D(stimulus, deltat=0.00005, v_zero=0.0, threshold=1.0, v_base=0.0,
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


def tau_ref_scan(taureflist, t, ntrials, params, stimulus, kernel):
    """
    Do a scan in membrane tau and refractory period in 1D integrate and fire neuron regarding the long time decay.
    
    Parameters
    ----------
    taureflist: 1-D array / list
        The list of values to be scanned for tau and refractory period
    t: 1-D array
        time in seconds
    ntrials: float
        Number of trials fo the peristimulus time histogram
    params: dictionary
        The model parameter dictionary
    stimulus: 1-D array
        The array containing stimulus values
    kernel: 1-D array
        Array of the convolution kernel
        
    Returns
    -------
    decaydf: Dataframe
        The dataframe of decay index for all scan matrix (refractory period and tau pairs)
    """
    decayIndex = np.zeros([len(taureflist),len(taureflist)]) #columns for tau, rows for refractory

    for idxt, tau in enumerate(taureflist): #tau
        for idxr, ref in enumerate(taureflist): #refractory
    
            convolvedspklist = np.zeros([t.shape[0],ntrials]) #initialized list of convolved spikes
            spiketrains = np.zeros([t.shape[0],ntrials]) #initialized list of spike trains
    
            params['mem_tau'] = tau
            params['ref_period'] = ref
            print('tau=%f ref=%f' %(tau, ref))
    
            for i in range(ntrials):
                params['v_zero'] = np.random.rand()
                v_mems, spiketimes = simulate_1D(stimulus, **params)
                
                convolvedspikes, spikearray = convolved_spikes(spiketimes, stimulus, t, kernel)
                
                convolvedspklist[:,i] = convolvedspikes
                spiketrains[:,i] = spikearray
            
            peristimulustimehist = np.mean(convolvedspklist, axis=1)
            decayidx = np.max(peristimulustimehist[(t<1) & (t>0.15)]) / np.max(peristimulustimehist[(t>=9) & (t<9.84995)])
            decayIndex[idxt, idxr] = decayidx
            
    
    decaydf = pd.DataFrame(decayIndex)
    return decaydf


def decibel_transformer(power):
    """
    Transform power to decibel (0 dB is the maximum value in power data)
    
    Parameters
    ----------
    power: 1-D array
        The array of power values to be transformed into decibel
        
    Returns
    -------
    dB: 1-D array
        Decibel transformed power
    """ 
    dB = 10.0*np.log10(power/np.max(power))   # power to decibel
    return dB

def power_spectrum(stimulus, spiketimes, t, kernel, nperseg):
    """
    Calculate power spectrum for given cell and stimulus
    
    Parameters
    ----------
    Stimulus: 1-D array
        The stimulus array
    spiketimes: 1-D array
        The array containing spike times
    t: 1-D array
        The time array
    kernel: 1-D array
        Array of the convolution kernel
    nperseg: float
        Power spectrum number of datapoints per segment
        
    Returns
    --------
    f: 1-D array
        The array of power spectrum frequencies
    p: 1-D array
        The array of frequency powers
    meanspkfr: float
        The average firing rate in Hz over the entire stimulus
    """   
    t_delta = t[1]-t[0]
    #run the model for the given stimulus and get spike times
    #spiketimes, spikeISI, meanspkfr = stimulus_ISI_calculator(cellparams, stimulus, tlength=len(t)*t_delta)
        
    convolvedspikes, spikearray = convolved_spikes(spiketimes, stimulus, t, kernel)
    
    meanspkfr = len(spiketimes)/(t[-1]-t[-0])
    
    f, p = welch(convolvedspikes[t>0.1], nperseg=nperseg, fs=1/t_delta)
    return f, p, meanspkfr


def power_spectrum_transfer_function(frequency, t, contrast, fAMs, kernel, nperseg, amp=1, **cellparams):
    """
    Calculate the transfer function for a given set of AM (amplitude modulation) frequencies and cell model
    
    Parameters
    ----------
    frequency: float
        The stimulus sine frequency
    t: 1-D array
        The time aray
    contrast: float
        The AM strength (can be between 0 and inf)
    fAMs: 1-D array
        The array containing the AM frequencies 
    kernel: 1-D array
        The kernel array for convolution
    nperseg: float
        The power spectrum nperseg variable
    amp: float
        The amplitude of the sinus wave, set to 1 by default.
    *stimulusparams: list
        List of stimulus parameters. 
    **cellparams: dictionary
        Dictionary containing parameters of the cell model
    
    Returns
    -------
    tfAMs: 1-D array
        The array containing the transfer function values for the given array of fAMs.
    """
    pfAMs = np.zeros(len(fAMs))
    for idx, fAM in enumerate(fAMs):
        stimulus = amp * np.sin(2*np.pi*frequency*t) * (1 + contrast*np.sin(2*np.pi*fAM*t))
        spiketimes = mod.simulate(stimulus, **cellparams)
        f, p, __ = power_spectrum(stimulus, spiketimes, t, kernel, nperseg)
        power_interpolator = interpolate(f, p)
        pfAMs[idx] = power_interpolator(fAM)
    tfAMs = np.sqrt(pfAMs)/contrast #transfer function value
    return tfAMs


def whitenoise(cflow, cfup, dt, duration, rng=np.random):
     """Band-limited white noise.

     Generates white noise with a flat power spectrum between `cflow` and
     `cfup` Hertz, zero mean and unit standard deviation.  Note, that in
     particular for short segments of the generated noise the mean and
     standard deviation can deviate from zero and one.

     Parameters
     ----------
     cflow: float
         Lower cutoff frequency in Hertz.
     cfup: float
         Upper cutoff frequency in Hertz.
     dt: float
         Time step of the resulting array in seconds.
     duration: float
         Total duration of the resulting array in seconds.

     Returns
     -------
     noise: 1-D array
         White noise.
     """
     # next power of two:
     n = int(duration//dt)
     nn = int(2**(np.ceil(np.log2(n))))
     # draw random numbers in Fourier domain:
     inx0 = int(np.round(dt*nn*cflow))
     inx1 = int(np.round(dt*nn*cfup))
     if inx0 < 0:
         inx0 = 0
     if inx1 >= nn/2:
         inx1 = nn/2
     sigma = 0.5 / np.sqrt(float(inx1 - inx0))
     whitef = np.zeros((nn//2+1), dtype=complex)
     if inx0 == 0:
         whitef[0] = rng.randn()
         inx0 = 1
     if inx1 >= nn//2:
         whitef[nn//2] = rng.randn()
         inx1 = nn//2-1
     m = inx1 - inx0 + 1
     whitef[inx0:inx1+1] = rng.randn(m) + 1j*rng.randn(m)
     # inverse FFT:
     noise = np.real(np.fft.irfft(whitef))[:n]*sigma*nn
     return noise


def cross_spectral_density(stimulus, spiketimes, t, kernel, nperseg, calcoherence=False):
    """
    Calculate power spectrum for given cell and stimulus
    
    Parameters
    ----------
    Stimulus: 1-D array
        The stimulus array
    spiketimes: 1-D array
        The array containing spike times
    t: 1-D array
        The time array
    kernel: 1-D array
        Array of the convolution kernel
    nperseg: float
        Power spectrum number of datapoints per segment
    calcoherence: logical
        If true, the coherence is also calculated for the given stimulus and model parameters.
    Returns
    --------
    f: 1-D array
        The array of power spectrum frequencies
    psr: 1-D array
        The array of cross spectral density power
    fcoh: 1-D array
        The array of frequencies for coherence
    gamma: 1-D array
        Coherence between stimulus and response (0-1, 1 means noiseless perfect linear system.)
    """
    t_delta = t[1]-t[0]
    #run the model for the given stimulus and get spike times
    #spiketimes, spikeISI, meanspkfr = stimulus_ISI_calculator(cellparams, stimulus, tlength=len(t)*t_delta)
        
    convolvedspikes, spikearray = convolved_spikes(spiketimes, stimulus, t, kernel)
        
    f, psr = csd(convolvedspikes[t>0.1], stimulus[t>0.1], nperseg=nperseg, fs=1/t_delta)
    if calcoherence == True:
        fcoh, gamma = coherence(convolvedspikes[t>0.1], stimulus[t>0.1], nperseg=nperseg, fs=1/t_delta)
        return f, psr, fcoh, np.sqrt(gamma)
    else:
        return f, psr


def response_response_coherence(stimulus, noise, spiketimes, t, kernel, nperseg, flow=None, fup=None):
    """
    Calculate response-response coherence for a given cell and RAM stimulus
    
    Parameters
    ----------
    stimulus: 1-D array
        The stimulus time series array
    noise: 1-D array
        The white noise array. The spikes are locked to this as stimulus fluctuates as much as the carrier EODf, making
        the coherence between stimulus-response close to zero.
    spiketimes: n-D array
        The list containing all trials of spike times
    t: 1-D array
        The time array
    kernel: 1-D array
        Array of the convolution kernel
    nperseg: float
        Power spectrum number of datapoints per segment
    flow: float
        The lower frequency cutoff
    fup: float
        The upper frequency cutoff
        
    Returns
    --------
    fcoh: 1-D array
        The array of frequencies for coherence
    gammarr: 1-D array
        The array of response-response coherence
    gammars: 1-D array
        The array of stimulus-response coherence
    """
    t_delta = t[1]-t[0]
    #run the model for the given stimulus and get spike times
    #spiketimes, spikeISI, meanspkfr = stimulus_ISI_calculator(cellparams, stimulus, tlength=len(t)*t_delta)
    if len(spiketimes)==2:    
        convolvedspikes1, spikearray1 = convolved_spikes(spiketimes[0], stimulus, t, kernel)
        convolvedspikes2, spikearray2 = convolved_spikes(spiketimes[1], stimulus, t, kernel)
        
        fcoh, gammarr = coherence(convolvedspikes1[t>0.1], convolvedspikes2[t>0.1], nperseg=nperseg, fs=1/t_delta)
        return fcoh, np.sqrt(gammarr)
    
    else:
        convolvedspikes = []   
        presps = [] #response powers
        csdsrr = [] #array of all response-response csd
        
        csdsrs = [] #array of all stimulus-response csd
        
        fs, ps = welch(noise[t>0.1], nperseg=nperseg, fs=1/t_delta)#white noise power spectrum
        
        for i in range(len(spiketimes)):
            convolvedspike, __ = convolved_spikes(spiketimes[i], stimulus, t, kernel)
            fr, pr = welch(convolvedspike[t>0.1], nperseg=nperseg, fs=1/t_delta)
            if i == 0:
                finterval = (fr>flow) & (fr<fup)
            convolvedspikes.append(convolvedspike[t>0.1])
            pr = np.array(pr)
            presps.append(pr[finterval])

            fcoh, prs = csd(convolvedspike[t>0.1], noise[t>0.1], nperseg=nperseg, fs=1/t_delta)            
            csdsrs.append(prs[finterval])
            
        convolvedspikes = np.array(convolvedspikes)
        presps = np.array(presps)
      
        for idx1 in range(convolvedspikes.shape[0]):
            if idx1==len(spiketimes)-1:
                continue
            for idx2 in np.arange(idx1+1, convolvedspikes.shape[0]):
                #print(idx1, idx2)
                fcoh, prr = csd(convolvedspikes[idx1,:], convolvedspikes[idx2,:], nperseg=nperseg, fs=1/t_delta)
                csdsrr.append(prr[finterval])
                
        gammarr = np.abs(np.mean(csdsrr, 0))**2 / np.mean(presps, 0)**2
        gammars = np.abs(np.mean(csdsrs, 0))**2 / (np.mean(presps, 0) * ps[finterval])
        return fcoh, np.sqrt(gammarr), gammars
                
                                
def homogeneous_population(npop, t, stimulus, cellparams, kernel):
    """
    Simulate the homogeneous population activity by summing the spike trains up and convolving them.
    
    Parameters
    ----------
    npop: float
        The population size
    t: 1-D array
        The time array for the stimulus
    stimulus: 1-D array
        The stimulus array
    cellparams: dictionary
        The dictionary containing the model parameters for the chosen neuron
    kernel: 1-D array
        The array for the convolution kernel
        
    Returns
    -------
    popact: npop-D array
        The spike raster of the population activity
    summedactconv: 1-D array
        The convolved summed population activity
    """
    popact = np.zeros([np.int(npop),len(stimulus)])
    for i in range(np.int(npop)):
        spiketimes = mod.simulate(stimulus, **cellparams)
        spikearray = np.zeros(len(stimulus)) 
        spikearray[(spiketimes//(t[1]-t[0])).astype(np.int)] = 1
        popact[i,:] = spikearray
    return popact


def heterogeneous_population(npop, t, stimulus, kernel):
    """
    Simulate the heterogeneous population activity by summing the spike trains up and convolving them.
    
    Parameters
    ----------
    npop: float
        The population size
    t: 1-D array
        The time array for the stimulus
    stimulus: 1-D array
        The stimulus array
    kernel: 1-D array
        The array for the convolution kernel
    
    Returns
    -------
    popact: npop-D array
        The spike raster of the population activity
    summedactconv: 1-D array
        The convolved summed population activity
    I_LB: float
        The lower bound information 
    """
    heteroidx = 0
    parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
    heteropop = np.random.randint(0, len(parameters), np.int(npop))#choose cells randomly from the parameters population
    cells = np.unique(heteropop)#unique cells inside the population
    popact = np.zeros([np.int(npop),len(stimulus)])
    for c, cell in enumerate(cells):
        cell, EODf, cellparams = parameters_dictionary_reformatting(c, parameters)
        for q in range(len(heteropop[heteropop==c])):#run the simulation as much as the given cell is in the population
            spktimes = mod.simulate(stimulus, **cellparams)
            spkarray = np.zeros(len(stimulus)) 
            spkarray[(spktimes//(t[1]-t[0])).astype(np.int)] = 1
            popact[heteroidx, :] = spkarray
            heteroidx += 1
    return popact
 
    

def lower_bound_info(summedactconv, stimulus, t, nperseg, cflow, cfup):
    """
    Calculate the lower bound info I_LB for a given population activity.
    
    Parameters
    ----------
    summedactconv: 1-D array
        The convolved summed population activity
    stimulus: 1-D array
        The stimulus array
    t: 1-D array
        The time array for the stimulus
    nperseg: float
        The nperseg value for calculating coherence
    cflow: float
        The lower limit of the RAM frequency interval 
    cfup: float
        The upper limit of the RAM frequency interval
        
    Returns
    -------
    I_LB: float
        The lower bound information 
    """
    #summed activity coherence
    fcohsum, gammasum = coherence(summedactconv[t>0.1], stimulus[t>0.1], nperseg=nperseg, fs=1/np.diff(t)[0])
    whtnoisefrange = [(fcohsum>cflow) & (fcohsum<cfup)]
    fcohsum = fcohsum[tuple(whtnoisefrange)]
    gammasum = gammasum[tuple(whtnoisefrange)]

    #Lower bound info:
    df = np.diff(fcohsum)[0] #integration step
    I_LB = -np.sum(np.log2(1-gammasum)) * df
    if np.isnan(I_LB) == True:
        print('I_LB is somehow not a number and is therefore set to zero!')
        I_LB == 0
    return I_LB


def response_calculator(contrasts, fAMs, cellparams, whitenoiseparams, kernel, nperseg, frequency, tlength, correct=False):
    """
    Calculate the model response for a given set of contrasts, fAMs and cell parameters
    
    Parameters
    ----------
    contrasts: 1-D array
        Contrast array for calculating responses
    fAMs: 1-D array
        Amplitude modulation frequency array for SAM
    cellparams: dictionary
        The dictionary containing cell model parameters
    whitenoiseparams: dictionary
        The dictionary containing white noise parameters
    kernel: 1-D array
        The array of kernel
    nperseg: float
        Power spectrum nperseg value
    frequency: float
        The carrier frequency of the SAM and RAM stimuli (typically EODf)
    tlength: float
        The length of the stimulus
    correct: boolean
        If True, the SAM contrast is corrected so that stimulus power matches that of RAM
    Returns
    -------
    SAMpowers: 1-D array
        Model responses to SAM stimuli
    RAMpowers: 1-D array
        Model responses to RAM stimuli
    """
    #response powers for RAM and SAM
    RAMpowers = []
    SAMpowers = []
    dt = cellparams['deltat']
    t = np.arange(0, tlength, dt)

    for cidx, contrast in enumerate(contrasts):
        print(cidx)
        #create white noise for different contrasts
        whtnoise = contrast * whitenoise(**whitenoiseparams)
        #RAM stimulus for the model
        tRAM = t[1:]
        whtstimulus = np.sin(2*np.pi*frequency*tRAM) * (1 + whtnoise)
        
        #model response to RAM stimulus   
        whtspiketimes = mod.simulate(whtstimulus, **cellparams)
        #RAM response power
        __, RAMpower, __ = power_spectrum(whtstimulus, whtspiketimes, tRAM, kernel, nperseg)
        RAMpowers.append(RAMpower)
        pfAMr = np.zeros(len(fAMs)) #power at fAM for response
        for findex, fAM in enumerate(fAMs):
            #print(findex)
            
            #create stimulus and calculate power at fAM for rectified stimulus
            if correct==True:
                correctionfactor = 0.1220904473654484 / np.sqrt(2.473) #SAM stimulus power correction factor setting SAM
                                                                       #and RAM stimuli powers equal.
            else:
                correctionfactor = 1
                
            #first number is AM sine wave power / SAM stimulus power (SAM_stimulus_check_power.py) 
            #second number is RAM power / AM sine wave power (SAM_stimulus_check_power.py)
            SAMstimulus = np.sin(2*np.pi*frequency*t) * (1 + correctionfactor*contrast*np.sin(2*np.pi*fAM*t))
            npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
            
            #model response to the SAM stimulus and power spectrum
            SAMspiketimes = mod.simulate(SAMstimulus, **cellparams)
            frSAM, prSAM, __ = power_spectrum(SAMstimulus, SAMspiketimes, t, kernel, npersegfAM)
                        
            #interpolate the response power at fAM, later to be used for the transfer function
            presp_interpolator = interpolate(frSAM, prSAM)
            pfAMr[findex] = presp_interpolator(fAM)
            
        SAMpowers.append(pfAMr)
        
    RAMpowers = np.array(RAMpowers)
    SAMpowers = np.array(SAMpowers)
    return SAMpowers, RAMpowers 
