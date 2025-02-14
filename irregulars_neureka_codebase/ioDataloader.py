
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from library import nedc

class IODataset(Dataset):

    def __init__(self, recording, config):

        self.config = config

        self.recording = recording

    def __len__(self):
        return self.recording.getDuration()*self.config.dataset.fs//self.config.dataset.window_size

    def _windowize(self, data, window_size, stride):
        data = np.array(data)
        data = data[:len(data) // window_size * window_size]
        data = data.reshape(-1, window_size)
        return data

    def _get_signals(self, idx):
        len_from = idx*self.config.dataset.window_size
        len_to = (idx+1)*self.config.dataset.window_size

        ch_names = self.recording.getChannelNames()
        sig = []
        for ch in range(len(ch_names)):
            sig.append(self.recording.getChannel(ch)[len_from:len_to])

        # Put signals in a montage
        (signals_ds, _) = nedc.rereference(sig, [x.upper() for x in ch_names])
        data = np.array(signals_ds).T

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data-mean)/(std+1e-8)

    def __getitem__(self, idx):
        signal = self._get_signals(idx)
        signal = self._windowize(signal, self.config.dataset.window_size, self.config.dataset.stride)

        return {"data":{"raw":signal},"idx": idx}

class IODataloader():

    def __init__(self, recording, config):
        """
        :param config:
        """
        self.config = config

        ds = self._get_dataset(recording, config)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0))-1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))
        
        self.loader = torch.utils.data.DataLoader(ds,
                                                    batch_size=self.config.training_params.test_batch_size,
                                                    shuffle=False,
                                                    num_workers=num_cores,
                                                    # collate_fn=collate_fn_padd,
                                                    pin_memory=self.config.training_params.pin_memory)
    def _get_dataset(self, recording, config):

        ds = IODataset(recording, config)

        return ds