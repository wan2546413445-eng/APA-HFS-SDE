import h5py
import numpy as np

map_file = "/mnt/SSD/wsy/fastmri_data/brain_multicoil_train/maps/file_brain_AXFLAIR_200_6002452.h5"

with h5py.File(map_file, "r") as f:
    smaps = f["s_maps"][:]

print(smaps.shape, smaps.dtype)
print("nan:", np.isnan(smaps).any())
print("inf:", np.isinf(smaps).any())
print("max:", np.abs(smaps).max(), "mean:", np.abs(smaps).mean())