# check_hfs_dataset_protocol.py
from configs.hfssde.ddpm_continuous import get_config
from utils.datasets import FastMRIKneeDataSet

config = get_config()

for mode in ["training", "test", "sample"]:
    print("\n==============================")
    print("checking mode:", mode)
    dataset = FastMRIKneeDataSet(config, mode)
    print("len(dataset):", len(dataset))

    for idx in range(10):
        print(dataset.debug_index(idx))