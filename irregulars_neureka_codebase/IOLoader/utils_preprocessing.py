from scipy import signal


def standardizeEEGChannelName(name):
    """ Standardize the EEG channel names from "EEG FP1-REF" to "Fp1" """
    name = name.lower()
    if "eeg" in name:
        name = name.split(" ")[1]
    if "-" in name:
        name = name.split("-")[0]
    return name

def pre_process(ch_data, fs_data, fs_resamp = 200, 
                butter_order = 4, fc_low = 0.5, fc_high = 100):
    """ Resample and filter given channel """
    
    ## Filtering : 0.5 - 100 Hz

    b, a = signal.butter(butter_order, fc_low/(fs_data/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(butter_order, fc_high/(fs_data/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    ## Powerline-interference filtering : 50 and 60 Hz

    b, a = signal.butter(butter_order, [59.5/(fs_data/2), 60.5/(fs_data/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)
    
    b, a = signal.butter(butter_order, [49.5/(fs_data/2), 50.5/(fs_data/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)


    ## Resampling to 200 Hz

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))

    return ch_data, fs_resamp