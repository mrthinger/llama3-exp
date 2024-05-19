import glob
import os
import torch
from tqdm import tqdm

dir = "model_weights/Smaug-Llama-3-70B-Instruct"
out_dir = f"{dir}/verify"
os.makedirs(out_dir, exist_ok=True)
paths = glob.glob(os.path.join(dir, "*.pth"))

for path in tqdm(paths):
    state_dict = torch.load(path)
    with open(f"{out_dir}/state_dict_stats_{os.path.basename(path)}.txt", "w") as f:
        for key, value in sorted(state_dict.items()):
            f.write(f"{key}: {value.shape}\n")
