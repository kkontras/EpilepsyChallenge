import os
import pyedflib
import warnings
from classes.utils import pre_process, standardizeEEGChannelName, extractMontage
import numpy as np

class Recording_Reader:
    def __init__(
        self,
        data,
        channels: tuple[str],
        fs: tuple[int],
        montage: str,
        rec_path: str
    ):
        """ Initiate an EEG instance

        Args:
            data (List(NDArray[Shape['1, *'], float])): a list of data arrays. Each channel's data is stored as an entry in the list as a data array that stores the samples in time.
            channels (tuple[str]): tuple of channels as strings.
            fs (tuple[int]): Sampling frequency of each channel.
            montage (str): the used montage
            rec_path (str): path to EDF file.
        """
        self.data = data
        self.preprocessed = False
        self.channels = channels
        self.montage = montage
        self.rec_path = rec_path

        # Reduce list of sampling frequencies to one sampling frequency
        if np.all(np.array(fs) != fs[0]):
            warnings.warn('Channels have different sampling frequencies for recording '+rec_path)
            self.fs = fs
        else:
            self.fs = fs[0]

    def preprocessData(self, american = False):
        """ Resample and filter the channels in this recording.
        If the preprocessing has already happened, then the channels are not affected. """

        # If the preprocessing has already happened, then the channels are not affected. 
        if self.preprocessed:
            return None
 
        # Resample and filter the channels
        res = []
        for i in range(len(self.getChannelNames())):
            tmp, newfs = pre_process(self.getChannel(i), self.getFs(), american)
            res.append(tmp)
        self.fs = newfs
        self.data = res
        self.preprocessed = True
        return None
    
    def getData(self):
        """ Get all channels """
        return self.data
    
    def getChannel(self, i):
        """ Get the channel with index i """
        return self.data[i]
    
    def getChannelNames(self):
        """ Get the names of all channels """
        return self.channels
    
    def getFs(self):
        """ Get the sampling frequdency """
        return self.fs
    
    def getMontage(self):
        """ Get the montage (referential vs ...) """
        return self.montage

    @classmethod
    def loadData(
        cls,
        rec_path: str
    ):
        """ Instantiate a data object from an EDF file.
        Non-EEG channels are discarded.
        Channel names are standardized.

        Args:
            rec_path (str): path to EDF file.
            
        Returns:
            Data: returns a Data instance containing the data of the EDF file.
        """

        # Extract the data from the file at the given location
        data = list()
        channels = list()
        samplingFrequencies = list()
        if os.path.exists(rec_path):
            with pyedflib.EdfReader(rec_path) as edf:
                samplingFrequencies.extend(edf.getSampleFrequencies())
                channels.extend(edf.getSignalLabels())
                n = edf.signals_in_file
                for i in range(n):
                    data.append(edf.readSignal(i))
                edf._close()
        else:
            warnings.warn('Recording ' + rec_path + ' could not be loaded!')

        # Remove non-EEG channels
        idxs = []
        for i in range(len(channels)):
            if 'eeg' not in channels[i].lower():
                idxs.append(i)
        for i in reversed(idxs):
            channels.pop(i)
            data.pop(i)
        
        # Extract the montage (given a channel name)
        montage = extractMontage(channels[0])

        # Standardize the channel names from "EEG F1-REF" to "F1"
        for i in range(len(channels)):
            channels[i] = standardizeEEGChannelName(channels[i])

        return cls(
            data,
            channels,
            samplingFrequencies,
            montage,
            rec_path
        )