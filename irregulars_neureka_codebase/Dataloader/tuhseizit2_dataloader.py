
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py
from library import nedc
from utils.iaaft import surrogates

class TUHSeizIT2Dataset(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode = mode

        if "sz2" in config.dataset.data_path.keys():
            self.h5_file_sz2 = h5py.File(config.dataset.data_path["sz2"], 'r', swmr=True)
        else:
            self.h5_file_sz2 = False
        if "tuh" in config.dataset.data_path.keys():
            self.h5_file_tuh = h5py.File(config.dataset.data_path["tuh"], 'r', swmr=True)
        else:
            self.h5_file_tuh = False

        #or load it from cumulated dict
        with open("/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/Dataloader/seizit2_compatible_patient_dict.pkl", "rb") as f:
            self.patient_dict_sz2 = pickle.load(f)

        with open("/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/Dataloader/tuh_patient_dict.pkl", "rb") as f:
            self.patient_dict_tuh = pickle.load(f)

        if self.h5_file_sz2 and self.h5_file_tuh:
            self.patient_dict = {**self.patient_dict_sz2, **self.patient_dict_tuh}
        elif self.h5_file_sz2:
            self.patient_dict = self.patient_dict_sz2
        elif self.h5_file_tuh:
            self.patient_dict = self.patient_dict_tuh
        #merge the two patient_dicts and on each key add the "sz2" or "tuh" as value
        self.patient_dataset = {patient:"sz2" if patient in self.patient_dict_sz2 else "tuh" for patient in self.patient_dict.keys()}

        self._subselect_patients(mode=mode)

        self._get_cumulative_lens()

    def __len__(self):
        return int(list(self.cumulated_dict.keys())[-1])



    def _discard_non_seizure(self):
        new_patient_dict = {}
        for dataset in self.patient_dict.keys():
            for patient in self.patient_dict[dataset].keys():
                    for session in self.patient_dict[dataset][patient].keys():
                        if len(self.patient_dict[dataset][patient][session]["events"]) == 0:
                            if random.random() < self.config.dataset.discard_full_prob:
                                continue

                        if dataset not in new_patient_dict.keys():
                            new_patient_dict[dataset] = {}
                        if patient not in new_patient_dict[dataset].keys():
                            new_patient_dict[dataset][patient] = {}
                        new_patient_dict[dataset][patient][session] = self.patient_dict[dataset][patient][session]

        print("We discard non-seizure recordings: {}".format(self.mode))
        print("Number of patients before: ", len(self.patient_dict.keys()))
        print("Number of patients after: ", len(new_patient_dict.keys()))
        self.patient_dict = new_patient_dict

    def _discard_non_seizure_windows(self):
        #estimate current ratio of seizure to non-seizure windows
        count = {"non_seizure": 0, "seizure": 0}
        for dataset in self.patient_dict.keys():
            for patient in self.patient_dict[dataset].keys():
                for session in self.patient_dict[dataset][patient].keys():
                    count["non_seizure"] += self.patient_dict[dataset][patient][session]["labels_per_window"].shape[0] - self.patient_dict[dataset][patient][session]["labels_per_window"].sum()
                    count["seizure"] += self.patient_dict[dataset][patient][session]["labels_per_window"].sum()
        current_ratio = count["seizure"]/count["non_seizure"]
        if current_ratio > self.config.dataset.ratio_sz_nsz:
            print("Current ratio is {} and desired ratio is {} so we discard nothing".format(current_ratio, self.config.dataset.ratio_sz_nsz))
            return
        #how many non seizure should we discard?
        to_discard = int((self.config.dataset.ratio_sz_nsz-current_ratio)*count["non_seizure"])
        discard_ratio = to_discard/count["non_seizure"]
        print("We discard randomly further {} non-seizure windows".format(to_discard))

        #discard non-seizure windows
        new_patient_dict = {}
        for dataset in self.patient_dict.keys():
            for patient in self.patient_dict[dataset].keys():
                for session in self.patient_dict[dataset][patient].keys():
                    if len(self.patient_dict[dataset][patient][session]["events"]) > 0:
                        if dataset not in new_patient_dict.keys():
                            new_patient_dict[dataset] = {}
                        if patient not in new_patient_dict[dataset].keys():
                            new_patient_dict[dataset][patient] = {}
                        new_patient_dict[dataset][patient][session] = self.patient_dict[dataset][patient][session]
                        discard_idx = np.zeros(len(new_patient_dict[dataset][patient][session]["labels_per_window"]))
                        for i in range(len(new_patient_dict[dataset][patient][session]["labels_per_window"])):
                            #check if previous or after are not seizures
                            if i > 0 and i < len(new_patient_dict[dataset][patient][session]["labels_per_window"])-1:
                                if new_patient_dict[dataset][patient][session]["labels_per_window"][i-1] == 0 and new_patient_dict[dataset][patient][session]["labels_per_window"][i+1] == 0:
                                    if random.random() < discard_ratio:
                                        discard_idx[i] = 1
                        new_patient_dict[dataset][patient][session]["discard_idx"] = discard_idx
                        new_duration = len(new_patient_dict[dataset][patient][session]["labels_per_window"]) - discard_idx.sum()

                        new_patient_dict[dataset][patient][session]["duration"] = (new_duration * self.config.dataset.window_size)/self.config.dataset.fs
        self.patient_dict = new_patient_dict

    def _subselect_patients(self, mode):

        patient_names = list(self.patient_dict.keys())
        #We decided for the challenge to skip the test set completely. If you need it change it accordingly
        # train_patients, test_patients = train_test_split(patient_names, test_size=0.01, random_state=42)
        train_patients, val_patients = train_test_split(patient_names, test_size=0.2, random_state=self.config.training_params.seed)
        test_patients = []

        if mode == "train":
            for patient in val_patients+test_patients:
                self.patient_dict.pop(patient)
        elif mode == "val":
            for patient in train_patients+test_patients:
                self.patient_dict.pop(patient)
        elif mode == "test":
            for patient in train_patients+val_patients:
                self.patient_dict.pop(patient)

        self._discard_non_seizure()
        if mode == "train":
            self._discard_non_seizure_windows()


    def check_for_discard(self, patient, session, recording):
        if len(self.patient_dict[patient][session][recording]["events"]) == 0:
            return True
        else:
            return False

    def _discard_non_seizure_seizit2(self):
        new_patient_dict = {}
        for patient in self.patient_dict.keys():
            for session in self.patient_dict[patient].keys():
                if len(self.patient_dict[patient][session]["events"]) > 0:
                    if patient not in new_patient_dict.keys():
                        new_patient_dict[patient] = {}
                    new_patient_dict[patient][session] = self.patient_dict[patient][session]
        print("We discard non-seizure recordings")
        print("Number of patients before: ", len(self.patient_dict.keys()))
        print("Number of patients after: ", len(new_patient_dict.keys()))
        self.patient_dict = new_patient_dict


    def _get_cumulative_lens(self):

        self.cumulated_dict = {}
        cum_idx = 0 #number of windows per patient
        for patient in self.patient_dict.keys():
            for session in self.patient_dict[patient].keys():
                for recording in self.patient_dict[patient][session].keys():
                    self.cumulated_dict[cum_idx] = {"patient": patient, "session": session, "recording": recording, "len": self.patient_dict[patient][session][recording]["duration"]*self.config.dataset.fs}
                    cum_idx += self.patient_dict[patient][session][recording]["duration"]*self.config.dataset.fs//self.config.dataset.window_size

    def _choose_patient_session_recording_len(self, idx):
        #find the cumulative idx that is right smaller than the idx
        for i, cum_idx in enumerate(list(self.cumulated_dict.keys())):
            if cum_idx > idx:
                break
        cum_idx = list(self.cumulated_dict.keys())[i-1]

        patient = self.cumulated_dict[cum_idx]["patient"]
        session = self.cumulated_dict[cum_idx]["session"]
        recording = self.cumulated_dict[cum_idx]["recording"]

        len_from = (idx-cum_idx)*self.config.dataset.window_size
        len_to = (idx-cum_idx+1)*self.config.dataset.window_size
        if len_to > self.cumulated_dict[cum_idx]["len"]:
            len_to = self.cumulated_dict[cum_idx]["len"]
        return {"patient": patient, "session": session, "recording": recording, "len_from": int(len_from), "len_to": int(len_to)}

    def _windowize(self, data, window_size, stride):
        data = np.array(data)
        data = data[:len(data) // window_size * window_size]
        data = data.reshape(-1, window_size)
        return data

    def _get_signals_seizit2(self, demographics, label):
        patient = demographics["patient"]
        session = demographics["session"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        augment_flag = label.sum() > 0 and self.mode == "train" and self.config.model.args.augment_with_surrogates
        ch_names = [x.decode() for x in self.h5_file_sz2['sz2'][patient][session]['channel_names'][()]]
        sig = []
        for ch in ch_names:
            if augment_flag:
                sig.append(surrogates(self.h5_file_sz2['sz2'][patient][session][ch][len_from:len_to], 1).squeeze(axis=0))
            else:
                sig.append(self.h5_file_sz2['sz2'][patient][session][ch][len_from:len_to])

        # Put signals in a montage
        (signals_ds, _) = nedc.rereference(sig, [x.upper() for x in ch_names])
        data = np.array(signals_ds).T

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data-mean)/(std+1e-8)

    def _get_label_seizit2(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        events = self.h5_file_sz2['sz2'][patient][session]['events'][()]

        #find the label of the corresponding segment
        if len(events) == 0:
            return torch.zeros(len_to-len_from), events
        else:
            window_start = len_from
            window_end = len_to
            duration = self.h5_file_sz2['sz2'][patient][session]['duration'][()]
            total_label = torch.zeros(int(duration*self.config.dataset.fs))

            for event in events:
                start = event[0]*self.config.dataset.fs
                end = event[1]*self.config.dataset.fs
                total_label[int(start):int(end)] = 1
            return total_label[window_start:window_end], events

    def _get_signals_tuh(self, demographics, label):
        patient = demographics["patient"]
        session = demographics["session"]
        recording = demographics["recording"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        augment_flag = label.sum() > 0 and self.mode == "train" and self.config.model.args.augment_with_surrogates
        ch_names = [x.decode() for x in self.h5_file_tuh[patient][session][recording]['channel_names'][()]]
        sig = []
        for ch in ch_names:
            if augment_flag:
                sig.append(surrogates(self.h5_file_tuh[patient][session][recording][ch][len_from:len_to], 1).squeeze(axis=0))
            else:
                sig.append(self.h5_file_tuh[patient][session][recording][ch][len_from:len_to])

        # Put signals in a montage
        (signals_ds, _) = nedc.rereference(sig, [x.upper() for x in ch_names])
        data = np.array(signals_ds).T

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data-mean)/(std+1e-8)

    def _get_label_tuh(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        recording = demographics["recording"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        events = self.h5_file_tuh[patient][session][recording]['events'][()]

        #find the label of the corresponding segment
        if len(events) == 0:
            return torch.zeros(len_to-len_from), events
        else:
            window_start = len_from
            window_end = len_to
            duration = self.h5_file_tuh[patient][session][recording]['duration'][()]
            total_label = torch.zeros(int(duration*self.config.dataset.fs))

            for event in events:
                start = event[0]*self.config.dataset.fs
                end = event[1]*self.config.dataset.fs
                total_label[int(start):int(end)] = 1
            return total_label[window_start:window_end], events

    def _fill_in_empty_events(self, events, times):
        events = np.array(events)
        for _ in range(times - len(events)):
            events = np.vstack((events, [0, 0]))
        return events

    def __getitem__(self, idx):

        demographics = self._choose_patient_session_recording_len(idx)
        if self.patient_dataset[demographics["patient"]] == "sz2":
            label, events = self._get_label_seizit2(demographics)
            signal = self._get_signals_seizit2(demographics, label)
            signal = self._windowize(signal, self.config.dataset.window_size, self.config.dataset.stride)
        elif self.patient_dataset[demographics["patient"]] == "tuh":
            label, events = self._get_label_tuh(demographics)
            signal = self._get_signals_tuh(demographics, label)
            signal = self._windowize(signal, self.config.dataset.window_size, self.config.dataset.stride)

        return {"data":{"raw":signal},"label": label, "idx": idx, "demographics":demographics}

class TUHSeizIT2_Dataloader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        train_dataset, valid_dataset, test_dataset = self._get_datasets()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0))-1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))
        #skip last batch

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=num_cores,
                                                        shuffle=True,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        generator=g,
                                                        # collate_fn=collate_fn_padd,
                                                        worker_init_fn=seed_worker,
                                                        drop_last=True
                                                        )
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_cores,
                                                        drop_last=True,
                                                        # collate_fn=collate_fn_padd,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=num_cores,
                                                       drop_last=True,
                                                       # collate_fn=collate_fn_padd,
                                                       pin_memory=self.config.training_params.pin_memory)
    def _get_datasets(self):

        train_dataset = TUHSeizIT2Dataset(config=self.config, mode="train")
        valid_dataset = TUHSeizIT2Dataset(config=self.config, mode="val")
        test_dataset = TUHSeizIT2Dataset(config=self.config, mode="test")

        return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    from easydict import EasyDict

    config = EasyDict()
    config.dataset = EasyDict()
    config.training_params = EasyDict()
    config.dataset.window_size = 4096
    config.dataset.stride = 4096
    config.dataset.fs = 200
    config.dataset.data_path = {"sz2": "/esat/biomeddata/mbhaguba/seizeit2.h5",
                                "tuh": "/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5"}
    config.training_params.len_sample = 4096*30
    config.training_params.fs = 200
    config.training_params.batch_size = 32
    config.training_params.test_batch_size = 32
    config.training_params.pin_memory = False
    config.training_params.num_workers = 6
    config.training_params.seed = 0


    dl = TUHSeizIT2_Dataloader(config)
    list_of_labels = []
    for i, batch in tqdm(enumerate(dl.train_loader), total=len(dl.train_loader)):
        # print(batch["data"]["raw"].shape)
        list_of_labels.append(batch["label"])
        if i == 1000:
            break
    print(torch.unique(torch.cat(list_of_labels).flatten(), return_counts=True))
    # for batch in dl.valid_loader:
    #     print(batch["data"]["raw"].shape)
    #     print(batch["label"].shape)
    #     break
    # for batch in dl.test_loader:
    #     print(batch["data"]["raw"].shape)
    #     print(batch["label"].shape)
    #     break