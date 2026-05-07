import os
import sys
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils import *
import pickle


class FastMRIKneeDataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIKneeDataSet, self).__init__()
        self.config = config

        if mode in ["train", "training"]:
            self.kspace_dir = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_train_knee/kspace/"
            self.maps_dir = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_train_knee/maps/"
            input_pkl = "/mnt/SSD/wsy/fastmri_data/knee/data_slice.pkl"

        elif mode == "test":
            self.kspace_dir = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/kspace/"
            self.maps_dir = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_test/maps/"
            input_pkl = "/mnt/SSD/wsy/fastmri_data/knee/data_slice.pkl"

        elif mode == "sample":
            self.kspace_dir = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_val/kspace/"
            self.maps_dir = "/mnt/SSD/wsy/fastmri_data/knee/multicoil_val/maps/"
            input_pkl = "/mnt/SSD/wsy/fastmri_data/knee/data_slice.pkl"

        elif mode == "photom":
            self.kspace_dir = "data/photom/kspace/"
            self.maps_dir = "data/photom/map/"
            input_pkl = "/mnt/SSD/wsy/fastmri_data/knee/data_slice.pkl"

        elif mode == "datashift":
            self.kspace_dir = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/multicoil_brain/kspace/"
            self.maps_dir = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/multicoil_brain/maps/"
            input_pkl = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/data_slice.pkl"

        else:
            raise NotImplementedError(f"Unknown dataset mode: {mode}")

        self.mode = mode
        self.file_list = get_all_files(self.kspace_dir)

        # HFS official-code style:
        #   mode == "sample" or "photom": use all slices
        #   mode == "training"/"test"/"datashift": discard first skip_first_slices
        self.skip_first_slices = int(getattr(self.config.data, "skip_first_slices", 6))
        self.apply_hfs_skip = self.mode not in ["sample", "photom"]

        print("[INFO] mode =", self.mode)
        print("[INFO] skip_first_slices =", self.skip_first_slices)
        print("[INFO] apply_hfs_skip =", self.apply_hfs_skip)
        print("[INFO] kspace_dir =", self.kspace_dir)
        print("[INFO] maps_dir =", self.maps_dir)
        print("[INFO] input_pkl =", input_pkl)

        valid_files = []
        bad_files = []

        for f in self.file_list:
            with h5py.File(f, "r") as hf:
                kshape = hf["kspace"].shape

            if (
                kshape[-2] < self.config.data.image_size
                or kshape[-1] < self.config.data.image_size
            ):
                bad_files.append((os.path.basename(f), kshape))
            else:
                valid_files.append(f)

        if len(bad_files) > 0:
            print("[ERROR] Files smaller than target crop size:")
            for name, shape in bad_files:
                print("  ", name, shape)
            raise RuntimeError(
                "Clean HFS baseline should not silently exclude files. "
                "Please fix the data split or use a separate non-baseline config."
            )

        self.file_list = valid_files
        print("valid files:", len(self.file_list))

        self.num_slices = np.zeros((len(self.file_list),), dtype=int)

        with open(input_pkl, "rb") as pkl_file:
            data_dict = pickle.load(pkl_file)

        for idx, file in enumerate(self.file_list):
            basename = os.path.basename(file)
            temp_path = os.path.join(self.kspace_dir, basename)
            print("Input file:", temp_path)

            if basename not in data_dict:
                raise KeyError(
                    f"{basename} is not found in {input_pkl}. "
                    f"Please regenerate data_slice.pkl for the current split."
                )

            n_slices = int(data_dict[basename])

            if self.apply_hfs_skip:
                self.num_slices[idx] = int(max(n_slices - self.skip_first_slices, 1))
            else:
                self.num_slices[idx] = int(n_slices)

        self.slice_mapper = np.cumsum(self.num_slices) - 1

        print("[INFO] num files =", len(self.file_list))
        print("[INFO] total slices after mode rule =", int(np.sum(self.num_slices)))
        print("[DATASET PROTOCOL]")
        print("  mode:", self.mode)
        print("  skip_first_slices:", self.skip_first_slices)
        print("  total_files:", len(self.file_list))
        print("  total_used_slices:", int(np.sum(self.num_slices)))
        print("  first_file:", os.path.basename(self.file_list[0]) if len(self.file_list) > 0 else None)
        print("  first_file_used_slices:", int(self.num_slices[0]) if len(self.num_slices) > 0 else None)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])

        slice_idx = (
            int(idx)
            if scan_idx == 0
            else int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        )

        # HFS official-code style:
        #   training/test/datashift -> slice_idx + 6
        #   sample/photom -> keep original slice_idx
        if self.apply_hfs_skip:
            slice_idx = slice_idx + self.skip_first_slices

        maps_file = os.path.join(
            self.maps_dir, os.path.basename(self.file_list[scan_idx])
        )
        with h5py.File(maps_file, "r") as data:
            maps_idx = data["s_maps"][slice_idx]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = crop(
                maps_idx,
                cropx=self.config.data.image_size,
                cropy=self.config.data.image_size,
            )
            maps_idx = np.squeeze(maps_idx, 0)
            maps = np.asarray(maps_idx)

        raw_file = os.path.join(
            self.kspace_dir, os.path.basename(self.file_list[scan_idx])
        )
        with h5py.File(raw_file, "r") as data:
            ksp_idx = data["kspace"][slice_idx]
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = crop(
                IFFT2c(ksp_idx),
                cropx=self.config.data.image_size,
                cropy=self.config.data.image_size,
            )
            ksp_idx = FFT2c(ksp_idx)
            ksp_idx = np.squeeze(ksp_idx, 0)

            if self.config.data.normalize_type == "minmax":
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
                ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)

            elif self.config.data.normalize_type == "std":
                minv = np.std(ksp_idx)
                if minv == 0:
                    raise ValueError(
                        f"std(kspace) is zero for file={raw_file}, slice_idx={slice_idx}"
                    )
                ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)

            elif self.config.data.normalize_type == "img_std":
                ksp_idx = np.expand_dims(ksp_idx, 0)
                ksp_idx = IFFT2c(ksp_idx)
                ksp_idx = normalize_l2(ksp_idx)
                ksp_idx = FFT2c(ksp_idx)
                ksp_idx = np.squeeze(ksp_idx, 0)

            else:
                raise ValueError(
                    f"Unknown normalize_type: {self.config.data.normalize_type}"
                )

            kspace = np.asarray(ksp_idx)

        return kspace, maps

    def debug_index(self, idx):
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        slice_idx = (
            int(idx)
            if scan_idx == 0
            else int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        )

        actual_slice_idx = slice_idx
        if self.mode != "sample" and self.mode != "photom":
            actual_slice_idx = slice_idx + self.skip_first_slices

        return {
            "global_idx": int(idx),
            "scan_idx": int(scan_idx),
            "file": os.path.basename(self.file_list[scan_idx]),
            "logical_slice_idx": int(slice_idx),
            "actual_h5_slice_idx": int(actual_slice_idx),
        }

    def __len__(self):
        return int(np.sum(self.num_slices))


def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)

    if config.data.dataset_name in ["fastMRI_knee", "fastMRI_brain"]:
        dataset = FastMRIKneeDataSet(config, mode)
    else:
        raise ValueError(f"Dataset {config.data.dataset_name} is not supported.")

    if mode in ["train", "training"]:
        print("[DEBUG] mode =", mode)
        print("[DEBUG] training.batch_size =", config.training.batch_size)
        print("[DEBUG] sampling.batch_size =", config.sampling.batch_size)

        data = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            drop_last=False,
        )

    else:
        from utils.utils import worker_init_fn

        data = DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            num_workers=0,
            drop_last=False,
        )

    print(mode, "data loaded")
    return data

