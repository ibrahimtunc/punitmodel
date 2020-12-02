# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:17:06 2020

@author: Ibrahim Alperen Tunc
"""
#Poisson spike train
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st

random.seed(1000)
EODf = 800 #frequency of EOD in Hz 

#Initialize parameters:
tEOD = np.arange(0,1000, 0.001) #time in EOD cycles, divide by EOD frequency to get the value in seconds
tms = tEOD*1000/EODf #time in ms

#Histogram kernel parameters
std_kernel = 0.005 #the kernel should be wider than temporal resolution -> std bigger tahn bins of tEOD
smooth_x = np.arange(0,10.0,0.01)

#Calculate the spikes
randomnums = np.random.rand(len(tEOD))
spikeidx = np.squeeze(np.where(randomnums<=0.2)) #indexes of the spikes aligned with t
spiketimes = tEOD[spikeidx] #in EOD cycles
ISIEOD = np.diff(spiketimes)
histkernel = st.gaussian_kde(ISIEOD, bw_method=std_kernel/np.var(ISIEOD))
smoothed_data = histkernel(smooth_x) #calculate the pdf for the given time window

fig, ax = plt.subplots(1,1)
ax.plot(smooth_x, smoothed_data)
ax.set_title('ISI histogram')
ax.set_xlabel('ISI [EOD period]')
ax.set_ylabel('Probability density')

              
"""
# scipy kde kernel density estimate
# bandwith = std_gaussian/np.var(ISIEOD) this is to pass the gaussian bandwidth you wanna pass
"""