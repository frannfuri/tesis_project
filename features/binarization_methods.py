import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def numpy_to_str_sequence(np_bin):
    '''Convert a numpy binary sequence to string binary sequence'''
    str_binary = "".join(list((np_bin.astype(int)).astype(str)))
    return str_binary

# 1) Median Binarization
def median_bin(signal):
    s_median = np.median(signal)
    binary_signal = np.zeros(len(signal))
    for i in range(len(signal)):
        if signal[i] >= s_median:
            binary_signal[i] = 1
    return binary_signal

def plot_median_binarization(signal, bin_signal):
    below_td = bin_signal==0
    plt.axhline(np.median(signal), linestyle='dashed', c='red', label='median')
    plt.plot(signal, c='black')
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.title('Median binarization of the signal')
    plt.legend(loc='best')
    plt.show()

# 2) Hilbert-envelopment binarization
def hilbert_envelop_bin(signal):
    s_envep = abs(hilbert(signal))
    s_envep_med = np.median(s_envep)
    bin_envep = np.zeros(len(signal))
    for i in range(len(s_envep)):
        if s_envep[i] >= s_envep_med:
            bin_envep[i] = 1
    return bin_envep


def plot_hilbert_envelop_binarization(signal, bin_signal):
    envelop = abs(hilbert(signal))
    below_td = bin_signal == 0
    plt.axhline(np.median(envelop), linestyle='dashed', c='red', label='median(abs(Hilbert))')
    plt.plot(signal, c='black')
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.plot(envelop, label='abs(Hilbert)')
    plt.title('Hilbert-envelopment binarization of the signal')
    plt.legend(loc='best')
    plt.show()

# 3) Hilbert-power binarization
def hilbert_power_bin(signal):
    s_h_power = abs(hilbert(signal)**2)
    s_h_power_med = np.median(s_h_power)
    bin_s_h_power = np.zeros(len(s_h_power))
    for i in range(len(s_h_power)):
        if s_h_power[i] >= s_h_power_med:
            bin_s_h_power[i] = 1
    return bin_s_h_power

def plot_hilbert_power_binarization(signal, bin_signal):
    s_h_power = abs(hilbert(signal)**2)
    below_td = bin_signal == 0
    plt.axhline(np.median(s_h_power), linestyle='dashed', c='red', label='median(abs(Hilbert)**2)')
    plt.plot(signal, c='black')
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.plot(s_h_power, label='abs(Hilbert**2)')
    plt.title('Envelopment-Power binarization of the signal')
    plt.legend(loc='best')
    plt.show()

# 4) Slope-sign binarization
def slope_sign_bin(signal):
    slope = np.concatenate((signal[1:], np.array([0]))) - signal
    bin_slope = np.zeros(len(slope))
    for i in range(len(slope)):
        if slope[i] >= 0:
            bin_slope[i] = 1
        else:
            bin_slope[i] = 0
    # remove the last value, because the slope is not good defined
    return bin_slope[:-1]

def plot_slope_sign_binarization(signal, bin_signal):
    slope = np.concatenate((signal[1:], np.array([0]))) - signal
    below_td = bin_signal == 0
    plt.plot(signal, c='black')
    # remove the last value, because the slope is not good defined
    signal = signal[:-1]
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.plot(slope[:-1], '*',label='slope', c='gray')
    plt.axhline(0, c='lightgray', linestyle='dashed')
    plt.title('Slope-sign binarization of the signal')
    plt.legend(loc='best')
    plt.show()

# 5) Slope-sign of Hilbert-envelopment binarization
def slope_hilb_envelop_bin(signal):
    s_envelop = abs(hilbert(signal))
    slope_envelop = np.concatenate((s_envelop[1:], np.array([0]))) - s_envelop
    bin_slope_envelop = np.zeros(len(slope_envelop))
    for i in range(len(slope_envelop)):
        if slope_envelop[i] >= 0:
            bin_slope_envelop[i] = 1
        else:
            bin_slope_envelop[i] = 0
    # remove the last value, because the slope is not good defined
    return bin_slope_envelop[:-1]

def plot_slope_hilb_envelop_binarization(signal, bin_signal):
    s_envelop = abs(hilbert(signal))
    slope_envelop = np.concatenate((s_envelop[1:], np.array([0]))) - s_envelop
    below_td = bin_signal == 0
    plt.plot(signal, c='black')
    # remove the last value, because the slope is not good defined
    signal = signal[:-1]
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.plot(slope_envelop[:-1], '*', label='Slope of abs(Hilbert)', c='gray')
    plt.axhline(0, c='lightgray', linestyle='dashed')
    plt.title('Slope-sign of Hilbert-envelopment binarization of the signal')
    plt.legend(loc='best')
    plt.show()

# 6) Slope-sign of Hilbert-power binarization
def slope_hilb_power_bin(signal):
    s_h_power = abs(hilbert(signal) ** 2)
    slope_h_power = np.concatenate((s_h_power[1:], np.array([0]))) - s_h_power
    bin_slope_h_power = np.zeros(len(slope_h_power))
    for i in range(len(slope_h_power)):
        if slope_h_power[i] >= 0:
            bin_slope_h_power[i] = 1
        else:
            bin_slope_h_power[i] = 0
    # remove the last value, because the slope is not good defined
    return bin_slope_h_power[:-1]

def plot_slope_hilb_power_binarization(signal, bin_signal):
    s_h_power = abs(hilbert(signal) ** 2)
    slope_h_power = np.concatenate((s_h_power[1:], np.array([0]))) - s_h_power
    below_td = bin_signal == 0
    plt.plot(signal, c='black')
    # remove the last value, because the slope is not good defined
    signal = signal[:-1]
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.plot(slope_h_power[:-1], '*', label='Slope of abs(Hilbert**2)', c='gray')
    plt.axhline(0, c='lightgray', linestyle='dashed')
    plt.title('Slope-sign of Hilbert-power binarization of the signal')
    plt.legend(loc='best')
    plt.show()

# 7) Mean binarization
def mean_bin(signal):
    s_mean = np.mean(signal)
    binary_signal = np.zeros(len(signal))
    for i in range(len(signal)):
        if signal[i] >= s_mean:
            binary_signal[i] = 1
    return binary_signal

def plot_mean_binarization(signal, bin_signal):
    below_td = bin_signal==0
    plt.axhline(np.mean(signal), linestyle='dashed', c='red', label='Mean')
    plt.plot(signal, c='black')
    plt.scatter(np.arange(len(signal))[~below_td], signal[~below_td], c='green')
    plt.scatter(np.arange(len(signal))[below_td], signal[below_td], c='blue')
    plt.title('Mean binarization of the signal')
    plt.legend(loc='best')
    plt.show()
