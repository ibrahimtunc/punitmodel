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
TODO:   .add mean fire rate in hist DONE
      
        .Normalize ISI axis in form of EODf DONE
      
        .Plot the sinus and spikes also in a second axis DONE
      
        .Make a model where probability to get a spike in a cycle is fixed number (0.2). For that take random number 
        generator (randn) and do logical indexing np.where(). Poisson spike train DONE in poisson_spike.py
      
        .Do amplitude modulation on the sinus with different contrasts (contrast is the same as amplitude modulation).
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
        
        .Do the steady state initial and baseline firing rate to all cells, and store the dataset for each cell in 
        separate panda dataframe csv file somehow. Also save the histogram values for each cell (use np.histogram to
        get the values and plot them separately, hist is in this script). Then in a separate script plot for each cell 
        the histogram and the firing rate curves (2 subplots so far), this to show the heterogeneity of the population.
        DONE
        
        .Do the amplitude modulation now with a sinusoidal, stimulus is EODf and contrast is sinusoidal with frequency
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
        array is the PSTH, plot together with SAM stimulus. DONE? You need to check the below point!
        
        .run each cell 10s with baseline stimulus, so it relaxes in the end, save the values v_dend, v_mem and
        adapt for each cell, which is then to be passed as initial conditions. take the last value for now and
        check if in SAM_stimulus_convolution.py the cell 10 still takes that long to adapt (in 10s). First do only
        for a_zero, not the others. If changing a_zero solves the adaptation problem, all is great. Create the new 
        models.csv with new a_zero. If new a_zero changes the f-I curve, there is problem. If not all is fine.
        v_zero is ok, because it is random and between 0-1. 
        another idea is to get rid of v_zero as cell parameter and instead to initialize it within the model stimulus
        function as np.random.rand(). comparison of the f-I curves by running the saving script for the new parameter
        set, then use the fI curve plot script to check what is up. For playing around with the stimulus model function
        take it to a new script (like helpers or something) DONE but not an issue of a_zero a_zero is +- in steady state
        
        !Problem not at a_zero, a_zero is already initialized in the fixed point!
        
        .Now reduce the integrate and fire model to 2d, leave only v_mem and adaptation (2D ODE):
        tau* dV/dt = -V -A + I(t) + D*noise (D is noise_strength)
        tau_A* dA/dt = -A
        if V>threshold -> V=V0 and A+=deltaA/tau_A
        The initial values: V0=0, threshold=1, tau=5ms, tau_A=50 ms, D=something in powers of 10.
        Then use amplitude modulation of the sine as input: amp*sin(2pi*f*t)+I_offset and play around until you get
        some similar activity. amp start with 1, f 10 Hz, Ioffset 2 (between 1 or 10)
        Then check again the peristimulus time histogram after convolution to see what is going on. DONE, see 
        integrate_and_fire_reduced_2D.py for the explanation. In short adaptation does not play a role in this long term
        decay of firing rate, but this occurs in the 2D model when refractory period is longer than single period of the
        stimulus and when noise is big enough that no phase locking to the stimulus happens. DONE        
        
        .Use reduced integrate and fire model, compare the spiking (voltage v_mem value) with the adaptation for a given
        time. For that, choose a specific time point, and do 100 trials and check for correlation between v_mem and
        adaptation value A. Do this also for different time points (beginning, middle, end, different phases of the sine
        like peak, through, negative etc etc.) DONE
        
        .integrate_and_fire_reduced_1D.py : for stimulus frequency of 500 (period 2 ms), take 10 values of refractory
        period and membrane tau from interval [0.2, 200] ms, take values logarithmically (use np.logspace). Then
        check the decay with the following index:
            decay_index = np.max(spike firing rate within 1st second) / np.max(spike firing rate within last second)
        this decay index will be your color code, you have 10x10 matrix of membrane tau and refractory period variable
        values. Create a color mesh based on those values and check if there is any regulatrity.
        np.logspace() linspace but logarithmic, but give powers of first and last values in base ten DONE but see below
        
        .Scan time constants/ref periods from 1 to 1000 ms and frequencies 1, 10, 100, 1000Hz. This will give clear 
        effects. You might want to reduce the resolution of you x and y axis to speed things up! (logspace step to 20) 
        DONE
        
        .alternative approach to above todo: find the cells with long time decay, and check in common parameter histogram
        if they cluster regarding some parameter values. for that plot the histograms and scatterplot the parameter 
        values of the cell models with long time decay. How does the issue with long decay translate to the p-unit 
        models? You checked the time constants and refactory periods. Are those models where t_ref is larger than tau_m 
        the ones that show the effect? Do for pairs of parameters to check for clusters. DONE
        
        .Do the above point by using the long time decay index you used for 1D integrate and fire model 
        decay_index = np.max(spike firing rate within 1st second) / np.max(spike firing rate within last second)
        DONE, 5 models stick out with mean decay index + 2*std, they do not seem that special
        
        .Check the decay index of the punit models with different time scales
        t_deltas = [10**-3, 10**-4, 5*10**-6].
        
        + What happens if you put the adaptation dynamics back into the I&F neuron? (maybe adjust the offset such that 
        in the steady state you have roughly the same average firing rate as without adaptation).

        .Check the speed: np.digitize vs 
                          spikearray[(spiketimes//t_deltat).astype(np.int)] = 1 vs 
                          spikearray, _ = np.histogram(spiketimes, t) ->timeit
                          DONE, best is second approach, makinfg the thing faster!
        
                
        +Check the integrate_and_fire_1d_reduced.py code whether the model simulation really gets the right parameters 
        (s versus ms - Benda did not find any flaw).
        
        .What happens if integration step gets larger by a factor of ten in the simulations?  Does the effect change in 
        the punit models if you change the integration time step (10 time smaller or larger)? DONE increasing the 
        integration step makes the effect to vanish, for smaller steps the effect is still there. BUT for bigger 
        integration steps the stimulus shows aliasing, although the sampling rate is well above Nyquist range (initial
        sampling frequency 20000.0 Hz, already for 2000.0 Hz (integration step 10x larger) the stimulus starts to look 
        weird.)
        
        +So now for the long time decay you know a few stuff, that e.g. the refractory period being longer than membrane
        tau can induce the long time decay in the reduced model, and increasing the integration step also causes 
        numerical problems (Euler method, increased time step decreases accuracy.). Rest is unclear so get rid of the 
        cells especially showing long time decay (use index, put a cutoff of e.g. 1.1 and use the rest of the cells)
        
        .Use the convolved spikes for a single trial (not the mean as in peristimulus time histogram), and create a 
        power spectrum for that (look again at the cricket scripts and use the code from there). In power spectrum, 
        you expect to see one big peak at EODf (should be sharp, play with npersec to get it right) and some other 
        peaks at f_AM and f_meanfr. Then, play around with AM frequency and see how the power spectrum value changes.
        Create a plot of power of f_AM as a function of f_AM, you expect to see bandpass filtering.
        For the display of power spectrum, you can also use dB (normalize with max value). DONE
        
        .Transfer function shows weird sinusoidal activity:
        Check if power spectrum peak at f_AM gets wider where the transfer function goes down. NO
        Check the same also for the stimulus (Peak at EODf and flankers (EODf+-f_AM)). NO
        If the peak gets wider when transfer function goes down, you should take a mean of multiple values
        around the peak. NOT THE CASE
        If the transfer function covaries with the power spectrum peaks of the stimulus, then it is a matter of 
        stimulus. CHECK BELOW
        punit_models_check_power_spectrum_and_transfer_function.py
        
        .Check transfer function for different stimulus amplitudes and contrasts DONE, EODf+-fAM shows fluctuation but in
        a very very tiny magnitude, I am not sure if that is what we are looking for, namely if the fluctuations seen at 
        those stimulus frequencies are nonlinearly amplified in the model response, or if something else in the model is 
        causing the fluctuations seen in the response transfer function.
        ask Jan Benda what is to be seen.
            
        *Cool facts on the transfer function:
            Transfer function H(w) = R(w)/S(w) where R and S are Fourier transformed response and stimulus functions
            Gain |H(w)| = |R(w)/S(w)| this is |R/S| if you disregard the phase shift introduced by the response function,
            and |R/S| corrseponds to A_r/A_s (with S(w) = A_s*delta(w-w_s) (delta is dirac function) as Fourier transform
            of simple sine wave is dirac delta. and R(w) = A_r*delta(w-w_s)*e^i*phi (additional phase shift)). In other
            words, the gain of the transfer function is the amplitudes of the stimulus and response divided. The power in
            the power spectrum is the square of those amplitudes (in Fourier space instead of variance you have power) 
            which is why you take square root of the response power. In the beginning, the fixed contrast was taken as
            the amplitude of the stimulus (hence divide sqrt(response power) with contrast), now you see the fluctuations
            in the power spectrum of the stimulus at EODf-fAM. So transfer function is now sqrt(p_response/p_stimulus)
            where you take the p_fAM for the response and p_EODF-fAM for the stimulus.
        
        .Check the stimulus power spectrum values -> if they show the fluctuations seen in transfer functions
        For that first make a function in helper_functions.py which takes the stimulus parameters (frequency, contrast, 
        fAM etc.) and generates the transfer function for the given cell. Then you can loop this over multiple contrasts
        and plot the transfer functions on top of each other in a plot. DONE, the stimulus indeed shows fluctuations at
        EODf+-fAM.
        
        .Now, check why the stimulus shows those fluctutations in the power spectrum. One reason could be that nperseg 
        does not exactly align with the stimulus AM frequency, so choose an nperseg which is a multiple of stimulus AM
        frequency. If that solves your problem thats cool. DONE, but nothing was solved, but the stimulus power spectrum
        became even more fluctuating.
        
        .Apart from the point above solving the problem, change the power spectrum of the stimulus to extract the
        envelope. To do that, take the absolute value of the stimulus and then do the power spectrum, this gives you
        the power at fAM and at EODf. Then use power at fAM to calculate the transfer function. DONE, all good
        
        *Another way of forming the transfer function:
            H(w) = R(w)/S(w) = R(w)*S_c(w)/(R(w)*S_c(w)) where S_c(w) is the complex conjugate of S(w)
            The denominator is now real, and equal to power of the stimulus, and the nominator is the cross power spectral
            density
            => H(w) = P_RS/P_SS where P_SS is the stimulus power and P_RS is the cross power spectral density.
        Calculate P_RS (use the function in https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html)
        P_RS is an array of complex numbers, containing the amplitude and the phase shift of the response function.
        To get to the gain (|H(w)|), take the real part of P_RS, and that also gives you the transfer function
        
        .You will use the above method for the white noise (see code provided by Jan Benda in email), first play
        around with that white noise to see its behavior in time and frequency domains, then give the white noise
        stimulus containing multiple frequencies to the model and check how the model behaves by using the transfer
        function. Calculate the transfer function by the method above. DONE, some issues asked in the email
        
        .You had a mistake on the white noise, the white noise stimulus should modulate the sine wave with EODf,
        similar to SAM. Correct that. Also look how sinus with EODf is modulated by white noise, for that create a 
        separate figure with RAM stimulus (cutoff frequency 50 Hz). For the transfer function, give the white noise power
        spectrum as stimulus input, because in the stimulus power spectrum the white noise frequencies are seen around
        EODf (as EODf+-white noise stimulus frequencies), whereas the response, locked to EODf, filters out the EODf 
        component from the stimulus and is only driven by the amplitude modulating sine wave. Because of that, you should
        either rectify the stimulus (which could cause complications when white noise interval gets around EODf) or simply
        use the white noise (contrast*white noise) as your stimulus input in the transfer function. Cheap solution but it
        works in the end. DONE, you still need to adjust the white noise stimulus contrast to match that of the SAM 
        stimulus (correction factor in-between, see Parseval theorem). Actually taking the abs(white noise stimulus) also
        matches the contrast between SAM and RAM, but other problems arise
        
        .Correct the unit of the transfer function gain: this is not power, but is Hz/mV, as the amplitude of the response
        (fire rate) is in Hz and that of the stimulus (voltage fluctuations) is in mV. You can also use Hz/% as unit. 
        Correct the figures. DONE new figures saved for cell 1.
        
        .Side quest: check why the power of the stimulus at AM fluctuates: you have before played with nperseg, but your
        mistake was making nperseg divisable by frequency not by period. Instead use this equation:
            nperseg = round(2**15*dt/T) * T/dt <=> nperseg = round(2**(15+log2(dt*f)) * 1/(dt*f) with 1/T = f = fAM
        This equation makes the nperseg divisable by period instead of frequency, first you check how many periods fit
        to 2**15 (by /T*dt), then round that value and mulitply back with T/dt to get to the value around 2**15
        DONE: Jan Benda's suggestion works like magic, power at fEOD for the stimulus fluctuates strongly, whereas the 
        fluctutation ceases for fAM, as nperseg is divisable for the fAM cycle but not for fEOD cycle
        
        *Parseval theorem: The variance of the stimulus in time domain with zero mean is calculated as:
            sigma^2 = 1/T * integral(s^2(t)dt) (lower and upper bounds -+T/2 respectively, s is the stimulus).
        Variance in frequency domain is equal to:
            sigma^2 = integral(P(w)dw) (lower and upper bounds fcutofflower and fcutoffupper, P is the stimulus power)
        
        Therefore, folllowing equality holds for same stimulus variance:
            1/T * integral(s^2(t)dt) = integral(P(w)dw)
        
        This means, for the SAM stimulus, the power spectrum has a much stronger peak at EDOF+-fAM (or at fAM when 
        stimulus is rectified), while for the white noise modulated stimulus the power spectrum shows weaker peaks, 
        because SAM stimulus has a few peaks at EODf and AM, while RAM (white noise modulated stimulus) has multiple
        peaks for all frequencies in the given interval, as the RAM has unit variance.
        
        .Based on the point above, check the power spectra of the SAM and RAM for different contrasts (no dB 
        transformation). DONE, the difference gets bigger with bigger contrast, which is expected, as increasing the 
        contrast increases the variance (by a factor of contrast^2), meaning that the integrals are also accordingly 
        scaled. The issue is, in SAM the power spectrum at fAM is so  narrow, that it can even be considered that 
        the power at fAM is scaled by contrast^2. On the other hand, for RAM the variance increase leads to the increase
        of power at all frequencies in fcutoff interval, but such that the integral of power within fcutoff is scaled by
        contrast^2, so each frequency component is effected less by this increase of power.
        
        +The linearity issue in contrast etc -> Jan Benda will think about that. Also replace the transfer function
        calculation there with the rectified stimulus power.
        
        .Plot the transfer function for different contrasts to both the RAM and the SAM. DONE
        
        .The transfer function gain was calculated by abs(csd/welch). Another similar metric used is called coherence
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html and below for the theoric
        explanation). Calculate coherence functions for different contrasts for both RAM and SAM. DONE for RAM
        
        *The main idea of coherence is as follows:
        For a linear system which converts the signal s to response r under some noise n, it can be written as:
            R = HS+N where the capital letters denote Fourier transformation and H is the transfer function.
            
        In the noiseless condition, the inverse operation of the linear system can revert the response to the stimulus:
            S = RH^-1 (where H^-1 is the inverse operation transfer function, think here the S is output and R input).
            
        In other words, if the system is linear and noiseless, H*H^-1 = (R/S)*(S/R) = 1
        
        In that regard, H = RS'/SS' and H^-1 = SR'/RR' (where ' denotes complex conjugate). Therefore, the coherence gamma
        can be defined as:
            gamma = H*H^-1 = (avg(RS') * avg(SR')) / (avg(SS') * avg(RR')) (avg is over time for each nperseg, as the 
            power is calculated for each frequency by averaging over nperseg times.)
            
            if gamma = 1, then the system is linear and noiseless. if gamma < 1, then the system is either linear and/or
            has noise in it. gamma cannot be bigger than 1. (If you are curious why, insert HS+N where you see R)
            
        .Error in last 2 figures in punit_models_transfer_function_SAM_RAM_stimulus_different_contrasts, the color code
        was inverted (ur a dumb dumb). Now all is correct just rerun the script and save accordingly. DONE
        
        -Update punit_models_transfer_function_SAM_RAM_stimulus_different_contrasts in such a way that it works for all 
        cells (do your waitforbuttonpress magic) DONE. Some cells show weird transfer functions on SAM stimulus
        
        .Conrast scaling between SAM and RAM: 
            -Approach 1: Numerical => use power_spectrum_SAM_RAM_different_contrasts.py, and divide the powers of 
            SAM and RAM stimuli for the same contrast. Then normalize this division with contrast squared (reason down
            below in the second approach). The values should be same for all contrasts. Then use this value as your 
            normalization factor between your SAM and RAM contrasts (to make sure which SAM contrast corresponds to 
            which RAM contrast, so that you can compare the transfer functions.). This is inaccurate, as for same 
            contrast the variance also scales the same (contrast^2), therefore division between SAM and RAM is constant
            over all contrasts, the value is around 67 (67.211 the mean over multiple fAMs, 
            code np.mean(np.mean(SAMstimpwr,1)/np.mean(whtnoisespwr[:,whtnoisefrange],1))). FAILED.
            -Approach 2: Numerical => White noise has unit variance
            signal variance sigma^2: sigma^2 = alpha*integral(P(w)dw) where alpha is some scaling factor. This is Parceval
            For RAM: contrast^2 = alpha*Pr*fc where fc is the cutoff frequency (f_high-f_low). This is correct because
            RAM has unit variance without contrast
            For SAM: contrast^2 = alpha*Ps*beta, where beta is the value encorporating the standard deviation of your SAM
            along with other stuff. Unlike RAM, there is no unit variance here, and because you rectify the stimulus and 
            make AM, calculating the stimulus variance is not so easy. Therefore, your analytical approach crumbles here 
            and boils down to another numerical approach.
            Plot the SAM power at fAM as a function of coherence (calculate power for a given fAM for different
            coherences). You expect to see a quadratic curve. From there you can calculate beta*alpha, which is 
            contrast^2/Ps. 
            If you want to have equal power for both SAM and RAM in frequencies (so if you want to have Pr=Ps):
                Pr = cr^2/(alpha*fc) = Ps = cs^2/(alpha*beta)
                <=> cr^2/fc = cs^2/beta <=> cs = cr*sqrt(beta/fc) where sqrt(beta/fc) is your correction factor
            This approach is DONE, works like MAGIC!
            Correction factor is 0.1220904473654484 (namely SAM and RAM stimulus power spectra are the same when SAM 
            contrast is generated by multiplying RAM contrast with this value)
          
        .Now check the cell model transfer functions with adjusted SAM contrasts to see if they align with RAM transfer 
        functions. Probably not as non-linearities in the system but still it is worth a shot to check. DONE, results are
        really weird. ALthough both SAM and RAM have the same power, the SAM transfer function has way higher gain, 
        implying that the cell model is driven way stronger for SAM than RAM. in SAM there is a single frequency component
        in AM, whereas in RAM there are multiple frequency components (the interval within fcutoff). This clearly shows
        the system nonlinearity (if you are not doing a programming miskate, which is of low probability), as only in
        nonlinear system the response can be influenced by the destructive interference between different frequency 
        components within the stimulus. Maybe first cell was werid (which i also highly suspect), or there are cells 
        which behave more linearly. Check also that.
            
        *Transfer functions: The transfer functions get smaller and smaller for bigger contrasts (Jan Benda paper
        sent on 19.10.2020 figure 6 shows the same for the responses), which is because with increase in contrast makes
        the system to saturate (compressive nonlinearity) and therefore response is not doubles with e.g. doubled stimulus
        
        *Coherence: Coherence increases with increased contrast, although due to the point above the system gets more non-
        linear with increase in contrast. The increase in coherence with contrast increase is due to the signal to noise
        ratio getting bigger (stimulus signal increases quadratically while neuronal model noise stays the same). You can
        therefore see that coherences stay +- the same for same contrasts. Furthermore, although for low frequencies your
        transfer function shows very small gain (the response is tiny to such stimuli), you have a very high coherence.
        This means most of the noise is also filtered out together with the signal, showing that the info transmission 
        of the cell with such stimuli is very strong.
        
        .Improvement on coherence. Decouple the coherence drop due to noise from coherence drop due to nonlinearity:
            Take response of the same model for 2 different RAM stimuli, in the end the non-linearity is the same and 
            the only difference in-between is the response noise in 2 different trials. Then run coherence on these 
            responses (response-response coherence gamma_RR). The square root of this coherence (sqrt(gamma_RR)) gives 
            you the upper bound. Plot this together with the stimulus response coherence. degree of sqrt(gamma_RR) 
            value differing from 1 gives you the degree of noise, the area between upper bound and stimulus response
            coherence gives you the degree of non-linearity. Code probably has error, as response response coherence is
            very low. This was because you calculated response response coherence by using different white noises which is
            completely wrong! Now it looks better, the system noise indeed gets smaller for bigger coherences due to 
            improved signal to noise ratio.
            
        +Population coding: Check the papers sent by Jan Benda (email, you need I_LB equation (lower bound information)).
        The population model is summed spike train. In that sense, the spikes of each neuron is summed together to form
        the response (i.e. sum up summa(dirac_delta(t-t_i)) for multiple neurons, in terms of coding this corresponds to
        summing up the spike trains of different neurons at same time bins). There will be 2 variants of the model, namely
        the homogeneous population (where each neuron is exactly the same) and the heterogeneous (neurons chosen randomly
        from the list of 75 models we have). In the homogeneous case, run the simulations many times for the same stimulus
        and the same model, each trial corresponds to a neuron. For the heterogeneous case run the simulations many times
        for the same stimulus but for different models. Sum up the spike trains in both cases to form the response. Then 
        calculate I_LB for different population sizes of n (n = 1,2,3,...) Create a plot of I_LB as a function of n, and
        do the x-axis (n) logarithmic (2,4,8,16 etc). This is the beginning of the model, then there are tons of stuff
        to decide on, like how to choose neurons in the heterogeneous case, whether to group them based on their baseline
        firing rate etc., also similar for the RAM stimulus (you will use RAM stimulus for all cases), of e.g. what should
        be the contrast, or if the cutoff frequencies should not go from 0 to 300 but like from 0 to 50 and then from 50 
        to 300 as a different RAM stimulus etc. (like whether to divide the frequency ranges to multiple intervals.). You
        will play around a lot with this model. Jan Benda sent you papers about the theoretical background of the model 
        assumptions (choose neurons randomly and sum up the spike trains -> convergence paper). Also read the locking 
        paper about the phase locking to EODf.
        
"""