# Adapted from STMC: src/normalizer.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/normalizer.py

import os
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        super().__init__()
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        mean = torch.load(self.mean_path, weights_only=True)
        std = torch.load(self.std_path, weights_only=True)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        # Ensure statistics live on the same device as the input tensor
        mean = self.mean if self.mean.device == x.device else self.mean.to(x.device)
        std = self.std if self.std.device == x.device else self.std.to(x.device)
        x = (x - mean) / (std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        # Ensure statistics live on the same device as the input tensor
        mean = self.mean if self.mean.device == x.device else self.mean.to(x.device)
        std = self.std if self.std.device == x.device else self.std.to(x.device)
        x = x * (std + self.eps) + mean
        return x
