from configs.hfssde.ddpm_continuous import get_config
from utils.datasets import get_dataset

config = get_config()
data = get_dataset(config, "test")

batch = next(iter(data))
print("x0 shape:", batch["x0"].shape)
print("gt shape:", batch["gt"].shape)
print("file_name:", batch["file_name"])
print("slice_idx:", batch["slice_idx"])
