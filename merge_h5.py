#I have in a folder a lot of h5 files that I want to merge into one h5 file.
#The h5 should have inside the following order: patient_id, recording_id, and then in there the data.

import h5py
import os
import numpy as np
from tqdm import tqdm
# Get all h5 files from the tsv file
path = "./TUHProcesser/dataset_lists/existingh5.tsv"
with open(path, 'r') as f:
    lines = f.readlines()
    h5_files = [line.split('\t')[-1].strip() for line in lines]

# Create the new h5 file
new_h5 = h5py.File("/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5", "w")

# Iterate over all h5 files
for h5_file in tqdm(h5_files, total=len(h5_files)):
    with h5py.File(h5_file, "r") as f:
        print(h5_file)
        name = h5_file.split("/")[-1].split("_")[0]
        patient_id = h5_file.split("/")[-1].split("_")[1]
        recording_id = h5_file.split("/")[-1].split("_")[2]
        # recording_id = h5_file.split("/")[-1].split("_")[2]
        # print(f.keys())
        # data = f
        # new_h5.create_dataset(f"{patient_id}/{recording_id}", data=data)
        #add each key of f in the new h5 file
        new_h5.require_group(f"{name}/{patient_id}/{recording_id}")
        #get the keys of the file
        print(new_h5[f"{name}/{patient_id}/{recording_id}"].keys())
        for key in f.keys():
            new_h5[f"{name}/{patient_id}/{recording_id}"].create_dataset(key, data=f[key])

        # print("Done")
# new_h5.close()
#open it againa and print the keys
print("We try to open it")
with h5py.File("/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5", "r") as f:
    print("We should have opened it")
    print(f.keys())
    for key in f.keys():
        print(key)
        for k in f[key].keys():
            print(k)
            for kk in f[key][k].keys():
                print(kk)
                print(f[key][k][kk].shape)
                print(f[key][k][kk])
                break
            break
        break





