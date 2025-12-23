from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def ensure_dirs():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(a, b):
    a = to_numpy(a).reshape(-1)
    b = to_numpy(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def savefig(name):
    ensure_dirs()
    out = Path("results/figures") / name
    plt.savefig(out, dpi=200, bbox_inches="tight")


def write_log(name, text):
    ensure_dirs()
    out = Path("results/logs") / name
    out.write_text(text)
