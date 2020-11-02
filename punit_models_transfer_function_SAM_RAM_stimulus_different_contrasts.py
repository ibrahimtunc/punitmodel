# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:15:07 2020

@author: Ibrahim Alperen Tunc
"""

import model as mod
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helpers
from scipy.signal import welch
from scipy.interpolate import interp1d as interpolate
import pandas as pd
import matplotlib as mpl
from cycler import cycler

#Compare transfer function of SAM and RAM stimuli.

#General parameters for both stimuli types
#cell and model parameters
parameters = mod.load_models('models.csv') #model parameters fitted to different recordings
contrasts = np.linspace(0,0.5,11)
contrasts[0] += 0.01
tlength = 100

cflow = 0
cfup = 300
#SAM parameters
fAMs = np.logspace(np.log10(1),np.log10(300),21)

for cell_idx in range(len(parameters)):
    cell, EODf, cellparams = helpers.parameters_dictionary_reformatting(cell_idx, parameters)
    dt = cellparams['deltat']

    #RAM white noise parameters
    whitenoiseparams = {'cflow' : cflow, #lower cutoff frequency
                        'cfup' : cfup, #upper cutoff frequency
                        'dt' : dt, #inverse of sampling rate
                        'duration' : tlength #in seconds
                        }
    locals().update(whitenoiseparams) #WOW this magic creates a variable for each dict entry!

    frequency = EODf
    t = np.arange(0, tlength, dt)
    
    #Calculate the stimuli
    whtnoises = np.zeros([len(t)-1,len(contrasts)])
    whtnoises2 = np.zeros([len(t)-1,len(contrasts)])
    whtnoisespwr = []
    SAMstimpwr = []
    nperseg = 2**12
    RAMtransferfuncs = []
    RAMcoherences = []
    SAMtransferfuncs = []
    gammarrs = [] #response response coherence
    #response powers for RAM and SAM
    RAMpowers = []
    SAMpowers = []
    #kernel parameters
    kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt}#kernel is muhc shorter for power spectrum
    
    #create kernel
    kernel, kerneltime = helpers.spike_gauss_kernel(**kernelparams)
    
    for cidx, contrast in enumerate(contrasts):
        print(cidx)
        #create white noise for different contrasts
        whtnoise = contrast * helpers.whitenoise(**whitenoiseparams)
        whtnoises[:,cidx] = whtnoise
        #calculate white noise power for different contrasts
        fwht, pwht = welch(whtnoise, fs=1/dt, nperseg=nperseg)
        whtnoisespwr.append(pwht)
        
        #RAM stimulus for the model
        tRAM = t[1:]
        whtstimulus = np.sin(2*np.pi*frequency*tRAM) * (1 + whtnoise)
       
        #model response to RAM stimulus   
        whtspiketimes = mod.simulate(whtstimulus, **cellparams)
        whtspiketimes2 = mod.simulate(whtstimulus, **cellparams)  #for response-response coherence       
        #cross spectral density and the transfer function for the RAM
        fcsdRAM, psrRAM, fcohRAM, gammaRAM = helpers.cross_spectral_density(whtnoise, whtspiketimes, tRAM, 
                                                                            kernel, nperseg, calcoherence=True)
        whttransferfunc = np.abs(psrRAM / pwht/2.473) #/2.473 corrects the RAM noise power to the stimulus power 
        RAMtransferfuncs.append(whttransferfunc)
        RAMcoherences.append(gammaRAM)
        
        #RAM response power
        __, RAMpower, __ = helpers.power_spectrum(whtstimulus, whtspiketimes, tRAM, kernel, nperseg)
        RAMpowers.append(RAMpower)
        #response-response coherence
        fcohrr, gammarr = helpers.response_response_coherence(whtstimulus, whtspiketimes, whtspiketimes2,
                                                              tRAM, kernel, nperseg)
        gammarrs.append(gammarr)
        #same thing as RAM for the SAM at different contrasts, except coherence thing is for now missing.
        #calculate for the given contrast each fAM stimulus and corresponding power
        pfAMs = np.zeros(len(fAMs)) #power at fAM for stimulus
        pfAMr = np.zeros(len(fAMs)) #power at fAM for response
        for findex, fAM in enumerate(fAMs):
            #print(findex)
            #create stimulus and calculate power at fAM for rectified stimulus
            correctionfactor = 0.1220904473654484 / np.sqrt(2.473) #SAM stimulus power correction factor setting SAM and
                                                                   #RAM stimuli powers equal.
            #first number is AM sine wave power / SAM stimulus power (SAM_stimulus_check_power.py) 
            #second number is RAM power / AM sine wave power (SAM_stimulus_check_power.py)
            SAMstimulus = np.sin(2*np.pi*frequency*t) * (1 + correctionfactor*contrast*np.sin(2*np.pi*fAM*t))
            npersegfAM = np.round(2**(15+np.log2(dt*fAM))) * 1/(dt*fAM) 
            fSAM, pSAM = welch(np.abs(SAMstimulus-np.mean(SAMstimulus)), fs=1/dt, nperseg=npersegfAM)
            pSAM_interpolator = interpolate(fSAM, pSAM)
            pfAMs[findex] = pSAM_interpolator(fAM)
            
            #model response to the SAM stimulus and power spectrum
            SAMspiketimes = mod.simulate(SAMstimulus, **cellparams)
            frSAM, prSAM, __ = helpers.power_spectrum(SAMstimulus, SAMspiketimes, t, kernel, npersegfAM)
            
            #interpolate the response power at fAM, later to be used for the transfer function
            presp_interpolator = interpolate(frSAM, prSAM)
            pfAMr[findex] = presp_interpolator(fAM)
    
        SAMstimpwr.append(pfAMs)
        SAMtransferfuncs.append(np.sqrt(pfAMr/pfAMs))
        SAMpowers.append(pfAMr)
        
    whtnoisespwr = np.array(whtnoisespwr)
    SAMstimpwr = np.array(SAMstimpwr)
    RAMtransferfuncs = np.array(RAMtransferfuncs)
    RAMcoherences = np.array(RAMcoherences)
    SAMtransferfuncs = np.array(SAMtransferfuncs)
    gammarrs = np.sqrt(np.array(gammarrs)) #square root is the response-response coherence
    RAMpowers = np.array(RAMpowers)
    SAMpowers = np.array(SAMpowers)
    
    fig, axts = plt.subplots(3,4, sharex=True, sharey='row')
    fig.suptitle('SAM and RAM transfer functions at different contrasts, cell %s' %(cell))
    lastax = axts[-1,-1]
    
    #remove last ax from sharey
    shay = lastax.get_shared_y_axes()
    shay.remove(lastax)
    #remove lastax as this will be used for legend
    lastax.remove()
    '''
    #In case you need the last ax again
    #create new yticks for lastax
    yticker = mpl.axis.Ticker()
    lastax.yaxis.major = yticker
    # The new ticker needs new locator and formatters
    yloc = mpl.ticker.AutoLocator()
    yfmt = mpl.ticker.ScalarFormatter()
    lastax.yaxis.set_major_locator(yloc)
    lastax.yaxis.set_major_formatter(yfmt)
    '''
    axts = np.delete(axts.reshape(12), 11)
    whtnoisefrange = (fwht>cflow) & (fwht<cfup) #frequency range to plot the power for white nose
    for idx, ax in enumerate(axts):
        ax.plot(fcsdRAM[whtnoisefrange], RAMtransferfuncs[idx, :][whtnoisefrange], 'k--', label='RAM')
        ax.plot(fAMs, SAMtransferfuncs[idx,:], 'r.-', label='SAM')
        ax.set_title('contrast=%.2f' %(contrasts[idx]))
        ax2=ax.twinx()
        ax2.plot(fcohRAM[whtnoisefrange], RAMcoherences[idx, whtnoisefrange])
        ax2.plot(fcohrr[whtnoisefrange], gammarrs[idx, whtnoisefrange], color='blue') 
        ax2.set_ylim([0, 1.0])
        if idx==7:
            ax2.set_ylabel('$coherence  \gamma$')
    axts[4].set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
    fig.text(0.45, 0.05, 'Frequency [Hz]')
    axts[-1].plot([], '-', color='blue', label='$\gamma_{rr}$')
    axts[-1].plot([], '-', color='#1f77b4', label='$\gamma_{sr}$')
    axts[-1].legend(loc='best', bbox_to_anchor=(1.8,1), prop={'size': 12})
    #lastaxyticks = np.linspace(0,1.1,12)
    #lastax.set_yticks(lastaxyticks)
    plt.subplots_adjust(wspace=0.3)
    
    #plot all RAM SAM transfer functions and coherences together
    RAMcols = plt.cm.Reds(np.linspace(0.2,1,len(contrasts)))  
    SAMcols = plt.cm.Blues(np.linspace(0.2,1,len(contrasts)))  
    cohcols = plt.cm.Greens(np.linspace(0.2,1,len(contrasts)))
    
    #RAM SAM transfer functions
    fig, axrsm = plt.subplots(1,1)
    axrsm.set_prop_cycle(cycler('color', RAMcols))
    axrsm.plot(np.tile(fcsdRAM[whtnoisefrange],[len(contrasts),1]).T, RAMtransferfuncs[:,whtnoisefrange].T)
    axrsm.set_prop_cycle(cycler('color', SAMcols))
    #axrsm.set_yscale('log')
    axrsm.plot(np.tile(fAMs,[len(contrasts),1]).T, SAMtransferfuncs.T, '.-')
    axrsm.set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
    axrsm.set_xlabel('Frequency [Hz]')
    axrsm.set_title('RAM and SAM stimulus transfer functions for different contrasts')
    fig.suptitle('cell %s'%(cell))

    #add colormaps
    ax2 = fig.add_axes([0.85, 0.25, 0.02, 0.5]) #The dimensions [left, bottom, width, height] 
    cmapRAM = mpl.colors.ListedColormap(RAMcols)
    
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(contrasts)+0.01)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmapRAM, norm=norm, 
                                    ticks=contrasts)
    cb1.set_label('RAM contrasts')
    ax3 = fig.add_axes([0.92, 0.25, 0.02, 0.5]) #The dimensions [left, bottom, width, height] 
    cmapSAM = mpl.colors.ListedColormap(SAMcols)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmapSAM, norm=norm, 
                                    ticks=contrasts)
    cb2.set_label('SAM contrasts')
    
    plt.subplots_adjust(left=0.06,bottom=0.07, right=0.845, top=0.92)
    
    #RAM transfer function and coherence
    fig, axrmc = plt.subplots(1,1)
    axrmc.set_prop_cycle(cycler('color', RAMcols))
    axrmc.plot(np.tile(fcsdRAM[whtnoisefrange],[len(contrasts),1]).T, RAMtransferfuncs[:,whtnoisefrange].T)
    axrmc2 = axrmc.twinx()
    axrmc2.set_prop_cycle(cycler('color', cohcols))
    #axrsm.set_yscale('log')
    axrmc2.plot(np.tile(fcsdRAM[whtnoisefrange],[len(contrasts),1]).T, RAMcoherences[:,whtnoisefrange].T)
    axrmc.set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
    axrmc2.set_ylabel('coherence  $\gamma$')
    axrmc.set_xlabel('Frequency [Hz]')
    axrmc.set_title('RAM transfer function and coherence for different contrasts')
    fig.suptitle('cell %s'%(cell))
    
    #add colormaps
    ax2 = fig.add_axes([0.85, 0.25, 0.02, 0.5]) #The dimensions [left, bottom, width, height] 
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmapRAM, norm=norm, 
                                    ticks=contrasts)
    cb1.set_label('RAM contrasts')
    
    ax3 = fig.add_axes([0.92, 0.25, 0.02, 0.5]) #The dimensions [left, bottom, width, height] 
    cmapcoh = mpl.colors.ListedColormap(cohcols)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmapcoh, norm=norm, 
                                    ticks=contrasts)
    cb2.set_label('coherence contrasts')
    
    plt.subplots_adjust(left=0.06,bottom=0.07, right=0.79, top=0.92)
    
    #plot the max RAM and SAM values as a function of contrast
    RAMmaxresponses = np.max(RAMpowers, 1)
    SAMmaxresponses = np.max(SAMpowers, 1)
    fig, axresponse = plt.subplots(1,1)
    axresponse.plot(contrasts, RAMmaxresponses, 'k-', label='RAM response')
    axresponse.plot(correctionfactor*contrasts, SAMmaxresponses, 'r-', label='SAM response')
    axresponse.set_title('Response power for different contrasts')
    axresponse.set_xlabel('Contrast')
    axresponse.set_ylabel('Power')
    axresponse.legend()
    
    #plot RAM and sam stim powers
    fig, axstimpowers = plt.subplots(3,4, sharex=True, sharey=True)
    axstimpowers = np.delete(axstimpowers.reshape(12), 11)
    for idx, ax in enumerate(axstimpowers):
        ax.plot(fwht[whtnoisefrange], whtnoisespwr[idx, whtnoisefrange], 'k--', label='RAM')
        ax.plot(fAMs, SAMstimpwr[idx,:], 'r.-', label='SAM')
    while True:
        if plt.waitforbuttonpress():
            plt.close('all')
            break
"""
#about to do something super lame:
for idx,tick in enumerate(lastaxyticks):
    fig.text(0.722, 0.105+0.02*idx, np.round(tick,4))
lastax.set_title('RAM coherences')
lastax.set_ylabel('Coherence factor $\gamma$')
lastax.yaxis.set_label_coords(-0.15, 0.5)
#Coherence increases with increase in contrast, meaning that for contrasts until 0.5, more contrast makes the system
#either more linear or less noisy. 
"""