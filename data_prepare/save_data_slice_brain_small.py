import os
import h5py
import pickle

directory = "/mnt/SSD/wsy/fastmri_data/brain_multicoil_train_small/kspace"
output_pkl = "/mnt/SSD/wsy/fastmri_data/brain_multicoil_train_small/data_slice.pkl"

h5_files = sorted([f for f in os.listdir(directory) if f.endswith(".h5")])

data_dict = {}

for h5_file in h5_files:
    full_path = os.path.join(directory, h5_file)

    with h5py.File(full_path, "r") as h5f:
        if "kspace" not in h5f:
            print("[SKIP no kspace]", h5_file)
            continue

        num_slices = h5f["kspace"].shape[0]
        data_dict[h5_file] = num_slices
        print(h5_file, num_slices)

with open(output_pkl, "wb") as pkl_file:
    pickle.dump(data_dict, pkl_file)

print(f"Data has been saved to {output_pkl}")
print("num files:", len(data_dict))
print("total slices:", sum(data_dict.values()))