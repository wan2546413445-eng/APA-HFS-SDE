import os
import pickle
import h5py
import numpy as np

TRAIN_KSPACE_DIR = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/multicoil_train_knee/kspace"
TEST_KSPACE_DIR = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/multicoil_test/kspace"
PKL_PATH = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/data_slice.pkl"

# 只检查 knee，忽略 brain
IGNORE_KEYWORDS = ["brain"]


def is_keep_file(filename: str) -> bool:
    name = filename.lower()
    return not any(k in name for k in IGNORE_KEYWORDS)


def list_h5_files(directory: str):
    files = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".h5") and is_keep_file(f):
            files.append(f)
    return files


def raw_num_slices(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["kspace"].shape[0])


def load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def used_slices_from_current_logic(mode: str, pkl_value: int) -> int:
    """
    完全模拟你当前 datasets.py 的逻辑：
    if mode != "sample" and mode != "photom":
        used = max(pkl_value - 6, 1)
    else:
        used = pkl_value
    """
    if mode != "sample" and mode != "photom":
        return max(int(pkl_value) - 6, 1)
    return int(pkl_value)


def summarize_split(split_name: str, kspace_dir: str, pkl_dict: dict):
    print("\n" + "=" * 80)
    print(f"[{split_name.upper()}] directory = {kspace_dir}")
    files = list_h5_files(kspace_dir)
    print(f"num files (after filtering brain): {len(files)}")

    header = (
        f"{'file':35s} | {'raw_h5':>6s} | {'pkl':>6s} | "
        f"{'train_used':>10s} | {'test_used':>9s} | {'sample_used':>11s}"
    )
    print(header)
    print("-" * len(header))

    total_raw = 0
    total_pkl = 0
    total_train = 0
    total_test = 0
    total_sample = 0

    for fname in files:
        h5_path = os.path.join(kspace_dir, fname)
        raw = raw_num_slices(h5_path)
        pkl_val = pkl_dict.get(fname, None)

        if pkl_val is None:
            print(f"{fname:35s} | {'MISSING':>6s} | {'MISSING':>6s}")
            continue

        train_used = used_slices_from_current_logic("training", pkl_val)
        test_used = used_slices_from_current_logic("test", pkl_val)
        sample_used = used_slices_from_current_logic("sample", pkl_val)

        total_raw += raw
        total_pkl += int(pkl_val)
        total_train += train_used
        total_test += test_used
        total_sample += sample_used

        print(
            f"{fname:35s} | {raw:6d} | {int(pkl_val):6d} | "
            f"{train_used:10d} | {test_used:9d} | {sample_used:11d}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':35s} | {total_raw:6d} | {total_pkl:6d} | "
        f"{total_train:10d} | {total_test:9d} | {total_sample:11d}"
    )

    # 帮你自动判断 pkl 更像哪种
    print("\n[Diagnosis]")
    if total_pkl == total_raw:
        print("data_slice.pkl 看起来存的是『原始总切片数』。")
        print("按当前 datasets.py，test/train 会再减 6，sample 不减。")
    elif total_pkl + 6 * len(files) == total_raw:
        print("data_slice.pkl 看起来存的是『已经减6后的切片数』。")
        print("这样会导致 sample 也和 test/train 一样少 6 张。")
    else:
        print("data_slice.pkl 既不像原始总切片数，也不像统一减6后的切片数。")
        print("说明它可能被多次混改过，需要重建。")

    return files


def inspect_global_index_mapping(files, kspace_dir, pkl_dict, mode="sample", max_show=15):
    """
    纯模拟 datasets.py 里的 global idx -> (scan_idx, slice_idx) 映射。
    不依赖 torch，不需要真正跑 dataloader。
    """
    print("\n" + "=" * 80)
    print(f"[Index Mapping Simulation] mode = {mode}")

    num_slices = []
    for fname in files:
        pkl_val = pkl_dict[fname]
        used = used_slices_from_current_logic(mode, pkl_val)
        num_slices.append(used)

    num_slices = np.array(num_slices, dtype=int)
    slice_mapper = np.cumsum(num_slices) - 1
    total_len = int(np.sum(num_slices))
    print(f"total dataset len in mode={mode}: {total_len}")

    show_n = min(max_show, total_len)
    for idx in range(show_n):
        scan_idx = int(np.where((slice_mapper - idx) >= 0)[0][0])
        slice_idx = (
            int(idx)
            if scan_idx == 0
            else int(idx - slice_mapper[scan_idx] + num_slices[scan_idx] - 1)
        )
        fname = files[scan_idx]
        print(
            f"global_idx={idx:3d} -> scan_idx={scan_idx:2d}, "
            f"file={fname}, slice_idx={slice_idx}"
        )


def main():
    print("Loading pkl:", PKL_PATH)
    pkl_dict = load_pkl(PKL_PATH)

    train_files = summarize_split("train", TRAIN_KSPACE_DIR, pkl_dict)
    test_files = summarize_split("test", TEST_KSPACE_DIR, pkl_dict)

    # 重点看你现在实际用于 ISTA/HFS 对比的 test knee 数据
    inspect_global_index_mapping(test_files, TEST_KSPACE_DIR, pkl_dict, mode="test", max_show=12)
    inspect_global_index_mapping(test_files, TEST_KSPACE_DIR, pkl_dict, mode="sample", max_show=12)


if __name__ == "__main__":
    main()
