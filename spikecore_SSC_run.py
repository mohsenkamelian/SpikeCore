# =============================================================================
# SpikeCore — Research Code
#
# Paper:
#   "Timing Is All You Need: SpikeCore, Learnable Delays,
#    and Gain Control for Neuromorphic Classification"
#   Advances in Neural Information Processing Systems (NeurIPS), 2026.
#
# Authors:
#   Mohsen Kamelian Rad, Sotiris Moschoyiannis, Roman Bauer
#
# Repository:
#   https://github.com/mohsenkamelian/SpikeCore
#
# Citation:
#   If you use this code in your research, please cite the accompanying paper.
#   See README.md for the full reference and BibTeX entry.
#
# File: spikecore_SSC_run.py
# Description:
#   Original SSC experiment script, including event preprocessing,
#   data augmentation, model construction, training, and evaluation.
#   The implementation and experiment settings are retained as used
#   in the authors' SSC experiments.
# =============================================================================

"""SpikeCore run for SSC / Neuromorphoic dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import tonic as tonic
from torchvision import datasets
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
from SpikeCore import SpikeCore, StepFunction
from backbone import feature_extractor

import copy

print("Loading dataset...")


# Training Loop
# ----------------------------------------------------------
# 1. Dataset Loading + Preprocessing
# ----------------------------------------------------------

n_time_steps = 50

        
class RandomFreqRoll:
    """Roll along frequency axis only — not time (too few bins)."""
    def __init__(self, max_shift=30, p=0.5):
        self.max_shift = max_shift
        self.p = p
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        shift = np.random.randint(-self.max_shift, self.max_shift)
        return np.roll(x, shift, axis=-1)

class CustomTimeStretch:
    def __init__(self, min_rate=0.8, max_rate=1.2):
        self.min_rate = min_rate
        self.max_rate = max_rate
    def __call__(self, events):
        events = events.copy()
        rate = np.random.uniform(self.min_rate, self.max_rate)
        events["t"] = (events["t"] * rate).astype(np.int64)
        return events
        
class RandomTimeRoll:
    def __init__(self, max_shift, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x):
        # x: [T, C] frame tensor
        if random.random() > self.p:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        return np.roll(x, shift, axis=0)

train_transform = transforms.Compose([
    CustomTimeStretch(0.8, 1.2),
    transforms.TimeJitter(std=2000),
    transforms.DropEvent(p=0.1),
    transforms.ToFrame(
        sensor_size=tonic.datasets.SSC.sensor_size,  # <-- changed
        n_time_bins=n_time_steps
    ),
    lambda x: x.astype("float32"),
    RandomFreqRoll(max_shift=20, p=0.3),
    RandomTimeRoll(max_shift=2, p=0.3),
    lambda x: x / (x.max() + 1e-6)
])

test_transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=tonic.datasets.SSC.sensor_size,  # <-- changed
        n_time_bins=n_time_steps
    ),
    lambda x: x.astype("float32"),
    lambda x: x / (x.max() + 1e-6)
])

# SSC has its own val split, so no need to carve one from train
train_ds = tonic.datasets.SSC(
    save_to="./data",
    split="train",                 
    transform=train_transform
)
val_ds = tonic.datasets.SSC(
    save_to="./data",
    split="valid",
    transform=test_transform
)
test_ds = tonic.datasets.SSC(
    save_to="./data",
    split="test",
    transform=test_transform
)
#train_ds = Subset(train_ds, range(6000))  # first 100 samples
#val_ds = Subset(val_ds, range(400))
#test_ds = Subset(test_ds, range(600))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) # num_workers and pin for speeding up
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)


# ----------------------------------------------------------
# 2. Buildin the model
# ----------------------------------------------------------
num_runs = 1        # For a test run. Increase for meaning over runs
f_test_accs = []

config = {
    "unit_type": SpikeCore,
    "unit_args": {
        "activation": StepFunction(),
        "g": F.leaky_relu,
        "decay": 0.9,
        "lambda_value": 0.0,
    },
    "N": [700, 512, 35],    # input_Size, HD size, and output size
    "lr": 0.001,
    "epochs": 200,
}

for run in range(num_runs):
    print(f"\n=========Run {run+1}/{num_runs}=======")
    print("Building Spiking Network...")
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #set_seed(42+run)      # Uncomment for multiple runs
    sizes = config["N"][1:]     # [256, 10]
    print(torch.cuda.is_available())
    net = feature_extractor(sizes, config["unit_args"], input_size=config["N"][0]).cuda()

    # ----------------------------------------------------------
    # 3. Loss & Accuracy (All PyTorch)
    # ----------------------------------------------------------

    def firing_rate_output(y_pred):
        """Mean firing rate across time dimension."""
        return y_pred.mean(dim=1)   # (batch, num_classes)

    # This is the loss function of SNU model
    def firing_loss(y_true, y_pred):
        fr = firing_rate_output(y_pred)
        y_true_oh = F.one_hot(y_true, num_classes=config["N"][-1]).float()
        return ((fr - y_true_oh) ** 2).sum(dim=1).mean()

    # This is our loss function for SpikeCore, more efficient on neuromorphic chips
    def firing_loss2(y, y_pred, spikes=None, delay_layers=None, delay_coeff=1e-4, spike_coeff=1e-3, target_spike_rate=0.04):
        # Base CE on mean firing rates
        fr = y_pred.mean(dim=1)
        ce = F.cross_entropy(fr + 1e-6, y, label_smoothing= 0.05)
        
        total_loss = ce
        
        # Spike rate regularization (L2 toward target sparsity; encourages low firing)
        if spikes is not None:
            mean_spike_rate = spikes.mean()  # scalar: avg spikes per timestep/neuron
            spike_reg = spike_coeff * (mean_spike_rate - target_spike_rate).pow(2)
            total_loss += spike_reg
        
        # Delay L1 regularization (sum of abs(delays) across all delay layers)
        if delay_layers is not None:
            delay_reg = 0.0
            for layer in delay_layers:
                # Use .all_delays if it exists, otherwise fall back to .delays
                if hasattr(layer, 'all_delays'):
                    delay_reg += sum(torch.abs(d).sum() for d in layer.all_delays)
                elif hasattr(layer, 'delays'):
                    delay_reg += torch.abs(layer.delays).sum()
            total_loss += delay_coeff * delay_reg
        
        return total_loss

    def accuracy(y_true, y_pred):
        fr = firing_rate_output(y_pred)
        pred = fr.argmax(dim=1)
        return (pred == y_true).float().mean().item()


    # ----------------------------------------------------------
    # 4. Training loop
    # ----------------------------------------------------------

    optimizer = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',             # 'max' because you track accuracy
    factor=0.5,             # or 0.1 / 0.3 — try 0.5 first for gentler drops
    patience=15,             # wait 5 epochs without improvement
    verbose=True,
    threshold=1e-4,         # small improvement threshold
    min_lr=1e-6             # don't go too low
    )
    scheduler4 = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["lr"],  # e.g. 1e-3
    epochs=config["epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.3,          # 30% warmup phase
    anneal_strategy='cos',
    div_factor=25,          # initial_lr = max_lr / 25: changed from 25 -> 10
    final_div_factor=1e4    # final_lr = initial_lr / 1e4 - (WAS 1e4)
    )

    scaler = GradScaler()

    def run_epoch(loader, train=True):
        net.train(train)
        total_loss = 0
        total_acc = 0
        total_correct = 0
        total_samples = 0
        #total_loss_val = 0
        #total_ce_val = 0  # To track actual classification progress
        #total_acc = 0
        count = 0
        criterion = nn.CrossEntropyLoss()#weight=class_weights)
        #delay_layers = [net.delay_layer]        # Modified loss

        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            y = y.long()
            h, _ = net(x)   # modified with delay reg
            #print(x.min().item(), x.max().item())
            loss = firing_loss2(y, h, spikes=h, delay_layers=None)  # modified with delay reg
            if train:
                optimizer.zero_grad()
                #scaler.scale(loss).backward()
                #scaler.step(optimizer)
                #scaler.update()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                #scheduler.step()
                #scheduler4.step()

            total_loss += loss.item() * x.size(0)
            total_acc += accuracy(y, h) * x.size(0)     # modified with delay reg
            count += x.size(0)
            # Statistics

        return total_loss / count, total_acc / count

    print("Training (AdamW)...")
    for epoch in range(config["epochs"]):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        test_loss, test_acc = run_epoch(test_loader, train=False)
        val_loss, val_acc = run_epoch(val_loader, train= False)
        print(f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f} | "
            f"Test Loss {test_loss:.4f}, Acc {test_acc:.4f}")
        scheduler2.step(val_acc)
        #scheduler3.step()


'''accs = np.array(f_test_accs)
N = accs.size

mean = accs.mean()
std = accs.std(ddof=1)          # sample standard deviation
ci95 = 1.96 * std / np.sqrt(N)

print(f"Mean accuracy: {mean:.4f}")
print(f"95% CI: [{mean - ci95:.4f}, {mean + ci95:.4f}]")
print(f"± {ci95:.4f}")
#np.save("accs_spikecoreSSC.npy", accs)'''

