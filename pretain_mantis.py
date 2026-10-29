import numpy as np
import pandas as pd 
import math
import torch
import torch.nn.functional as F
from mantis.architecture import Mantis8M
from mantis.architecture import MantisV2   
from mantis.trainer import MantisTrainer
from sklearn.utils.class_weight import compute_class_weight
import os

from sklearn.preprocessing import StandardScaler

from numpy.lib.stride_tricks import sliding_window_view

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = 'mantis_checkpoints'
os.makedirs(folder_path, exist_ok=True)

window_list = [7,14,30]

CHECKPOINTS = {
    "mantis":      (Mantis8M, "paris-noah/Mantis-8M", None),
    "mantis_plus": (Mantis8M, "paris-noah/MantisPlus", None),
    "utica":       (Mantis8M, "fegounna/Utica", "3cff4f954191b5bf9839b7a41117e2b24e7693ab"),
    "mantis_v2":   (MantisV2, "paris-noah/MantisV2", None),
    
}

result_checkpoint_list = ['mantis_7d_continued.pt',
 'mantis_7d_scratch.pt',
 'mantis_plus_7d_continued.pt',
 'utica_7d_continued.pt',
 #'utica_7d_scratch.pt',
 'mantis_v2_7d_continued.pt',
 'mantis_v2_7d_scratch.pt',
 'mantis_14d_continued.pt',
 'mantis_14d_scratch.pt',
 'mantis_plus_14d_continued.pt',
 #'mantis_plus_14d_scratch.pt',
 'utica_14d_continued.pt',
 #'utica_14d_scratch.pt',
 'mantis_v2_14d_continued.pt',
 'mantis_v2_14d_scratch.pt',
 'mantis_30d_continued.pt',
 'mantis_30d_scratch.pt',
 'mantis_plus_30d_continued.pt',
 #'mantis_plus_30d_scratch.pt',
 'utica_30d_continued.pt',
 #'utica_30d_scratch.pt',
 'mantis_v2_30d_continued.pt',
 'mantis_v2_30d_scratch.pt']

# Load data
ntl_cities = pd.read_csv('peaceful_cities/ntl_cities.csv').drop(columns= ['ntl', 'ntl_valid_frac','ntl_missing'])
ntl_cities['time'] = pd.to_datetime(ntl_cities['time'])
ntl_cities.rename(columns={'city_id':'case_id'}, inplace=True)

# Clean data
ntl_cities = ntl_cities.groupby('case_id').filter(lambda group: not group.isna().any().any()) ## remove these from cases
ntl_cities = ntl_cities.sort_values(['case_id','time'])
def is_consecutive(s):
    s = s.sort_values()
    # no duplicates, and every gap is exactly 1 day
    return s.is_unique and (s.diff().dropna() == pd.Timedelta(days=1)).all()

valid = ntl_cities.groupby("case_id")["time"].apply(is_consecutive)
ntl_cities = ntl_cities[ntl_cities["case_id"].isin(valid[valid].index)]

# keep cities with at least 60 days of data
ntl_cities['time_step'] = ntl_cities.groupby('case_id').cumcount()
n_days = ntl_cities.groupby('case_id')['time_step'].transform('size')
ntl_cities = ntl_cities[n_days >= 60]


def make_windows_fast(df, id_col="case_id", date_col="time", value_col="ntl_filled",
                      window=7, step=1):
    out = []
    for case_id, g in df.sort_values([id_col, date_col]).groupby(id_col):
        if len(g) < window:
            continue
        v = sliding_window_view(g[value_col].to_numpy(), window)[::step]
        d = g[date_col].to_numpy()
        starts = d[: len(v) * step : step]
        ends = d[window - 1 :][: len(v) * step : step]
        w = pd.DataFrame(v, columns=[f"t{i}" for i in range(window)])
        w.insert(0, "window_end", ends)
        w.insert(0, "window_start", starts)
        w.insert(0, id_col, case_id)
        out.append(w)
    return pd.concat(out, ignore_index=True)

def build_net(name, from_released):
    cls, ckpt, rev = CHECKPOINTS[name]
    net = cls(device=device)
    if from_released:
        net = net.from_pretrained(ckpt, revision=rev) if rev else net.from_pretrained(ckpt)
    return net

STEP = 7  # larger step = fewer near-duplicate windows

for window in window_list:
    win_len = window * 2

    windows = make_windows_fast(ntl_cities, window=win_len, step=STEP)
    meta = windows[["case_id", "window_start", "window_end"]].copy()
    X = windows.drop(columns=["case_id", "window_start", "window_end"]).to_numpy(dtype=float)

    sd = X.std(axis=1)
    keep = sd > 1e-8
    print(f"len {win_len}: dropping {(~keep).sum()} of {len(X)} windows")
    X, sd, meta = X[keep], sd[keep], meta[keep].reset_index(drop=True)
    meta.to_csv(f"{folder_path}/meta_{win_len}.csv", index=False)

    mu = X.mean(axis=1, keepdims=True)
    X_std = ((X - mu) / sd[:, None])[:, None, :]
    X_512 = F.interpolate(torch.tensor(X_std, dtype=torch.float32),
                          size=512, mode="linear", align_corners=False).numpy()

    for name in CHECKPOINTS:
        for mode, from_released in [("continued", True), ("scratch", False)]:
            num_epochs = 100
            file_name = f"{name}_{window}d_{mode}.pt"
            file_path = f"{folder_path}/{file_name}"
            
            if not(file_name in os.listdir(folder_path)) and file_name in result_checkpoint_list:
                print(f'Creating {file_name}')
                net = build_net(name, from_released)
                model = MantisTrainer(device=device, network=net)
                model.pretrain(
                    X_512,
                    num_epochs=num_epochs,
                    batch_size=128,
                    base_learning_rate=2e-3,
                    data_parallel=False,
                    learning_rate_adjusting=True,
                    file_name=file_path,
                )
                del model, net
                torch.cuda.empty_cache()
            else:
                print(f'Skipping {file_name}')