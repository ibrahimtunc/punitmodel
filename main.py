
import numpy as np
import matplotlib.pyplot as plt

from model import simulate, load_models

"""
Dependencies:
numpy 
matplotlib
numba (optional, speeds simulation up: pre-compiles functions to machine code)
"""


def main():
    # tiny example program:

    example_cell_idx = 20

    # load model parameter:
    parameters = load_models("models.csv")

    model_params = parameters[example_cell_idx]
    cell = model_params.pop('cell')
    EODf = model_params.pop('EODf')
    print("Example with cell:", cell)

    # generate EOD-like stimulus with an amplitude step:
    deltat = model_params["deltat"]
    stimulus_length = 2.0  # in seconds
    time = np.arange(0, stimulus_length, deltat)
    # baseline EOD with amplitude 1:
    stimulus = np.sin(2*np.pi*EODf*time)
    # amplitude step with given contrast:
    t0 = 0.5
    t1 = 1.5
    contrast = 0.3
    stimulus[int(t0//deltat):int(t1//deltat)] *= (1.0+contrast)

    # integrate the model:
    spikes = simulate(stimulus, **model_params)

    # some analysis an dplotting:
    freq = calculate_isi_frequency(spikes, deltat)
    freq_time = np.arange(spikes[0], spikes[-1], deltat)

    fig, axs = plt.subplots(2, 1, sharex="col")

    axs[0].plot(time, stimulus)
    axs[0].set_title("Stimulus")
    axs[0].set_ylabel("Amplitude in mV")

    axs[1].plot(freq_time, freq)
    axs[1].set_title("Model Frequency")
    axs[1].set_ylabel("Frequency in Hz")
    axs[1].set_xlabel("Time in s")
    plt.show()
    plt.close()


def calculate_isi_frequency(spiketimes, t):
    """
    Calculate ISI frequency
       
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
            return freqtime, t
        freqtime[tbegin:tend] = freq[i]
    freqtime[tend:] = freq[i]
    return freqtime

if __name__ == '__main__':
    main()
