import numpy as np

def sliding_windowing(complete_record, w_len, overlap, fs):
    ''' Function that receive a signal, and returns sliding windows of that signal.
    :param w_len: in SECONDS.
    :param overlap: a decimal number between 0 and 1.
    '''
    w_len = int(w_len*fs)
    N = len(complete_record)
    step = int(w_len - overlap*w_len)
    num_w = int(np.floor((N-overlap*w_len)/step))
    signal_windows = np.zeros((num_w, w_len))
    for i in range(num_w):
        signal_windows[i,:] = complete_record[i*step:i*step+w_len]
    return signal_windows

