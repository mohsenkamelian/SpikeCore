# =============================================================================
# SpikeCore — Research Code
#
# Paper:
#   "Timing Is All You Need: SpikeCore, Learnable Delays,
#    and Gain Control for Neuromorphic Classification"
#   Advances in Neural Information Processing Systems (NeurIPS), 2026 - Main Track.
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
# File: SpikeCore.py
# Description:
#   PyTorch implementation of the SpikeCore spiking unit, including learnable
#   membrane decay, spike-history-dependent threshold modulation, and
#   sigmoid-based surrogate gradients.
# =============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class StepFunction(nn.Module):
    def forward(self, x):
        sig = torch.sigmoid(x)
        hard = (x > 0).float()
        return sig + (hard - sig).detach()
# ----------------------------
# SpikeCore and its modifications
# ----------------------------
class SpikeCore(nn.Module):
    def __init__(self, num_units, activation=None, g=F.leaky_relu, decay=0.8, initVth=1.0, lambda_value=0.0, recurrent= False):
        super().__init__()
        self.num_units = num_units
        self.activation = StepFunction() if activation is None else activation
        self.g = g
        self.decay_init = decay
        self.initVth = initVth
        self.built = False
        self.recurrent = recurrent

    def build(self, input_size):
        self.kernel = nn.Parameter(torch.empty(input_size, self.num_units))
        nn.init.xavier_uniform_(self.kernel)
        # Bias acts as the baseline negative threshold
        self.bias = nn.Parameter(torch.full((self.num_units,), -self.initVth))
        
        if self.recurrent:
            self.recurrent_kernel = nn.Parameter(
                torch.empty(self.num_units, self.num_units) * 0.005)  # start very small!

        # Making decay learnable but bounded (0,1) via sigmoid is often better
        #self.register_buffer('decay', torch.full((self.num_units,), self.decay_init))  # Non-Trainable decay
        
        initial_values = torch.empty(self.num_units).uniform_(0.7, 0.95)
        self.decay = nn.Parameter(torch.log(initial_values / (1.0 - initial_values)))
        
        self.alpha_raw = nn.Parameter(torch.zeros(self.num_units))
        self.built = True

    @property
    def alpha(self):
        return 0.1 * torch.sigmoid(self.alpha_raw)

    def forward_sequence(self, x, state):
        if not self.built:
            self.build(x.shape[-1])

        batch, T, _ = x.shape
        Vm, _, adapt = state
        #Vm, spike_prev, adapt = state   # we now use spike_prev for recurrence
        
        # Pre-calculate the spatial projection for all time steps (efficiency)
        inp_term = x @ self.kernel 
        
        out_seq = []
        vm_seq = []   # added for Flexible readout
        for t in range(T):
            # 1. Recurrent update: Vm_t = decay * Vm_{t-1} + input_t
            # We use sigmoid on decay to keep it stable between 0 and 1
            d = torch.sigmoid(self.decay)
            
            # === RECURRENT TERM ==
            '''if self.recurrent:
                rec_term = spike_prev @ self.recurrent_kernel   # [batch, num_units]
                rec_term = rec_term / (self.num_units ** 0.5 + 1e-6)   # scale like attention
            else:
                rec_term = 0.0'''
                
            Vm = Vm * d + inp_term[:, t, :]    # Trainable decay
            #Vm = Vm * d + inp_term[:, t, :] + rec_term  # Trainable decay + recurrent 
            #Vm = Vm * self.decay + inp_term[:, t, :]   # non-trainable decay
            Vm = self.g(Vm) # Non-linearity on the membrane potential

            # 2. Dynamic Threshold: Bias + alpha * adaptation_state
            # Note: bias is negative, so Vm + bias is effectively Vm - Vth
            current_threshold = self.bias + (self.alpha * adapt)
            
            # 3. Generate Spike
            spikes = self.activation(Vm + current_threshold)   # My SpikeCore
            #spikes = self.activation(Vm + self.bias)        # SNU
            
            # 4. Update Adaptation State (decaying trace of past spikes)
            adapt = 0.9 * adapt + spikes.detach() 
            
            # 5. Reset (Optional: subtracts threshold or sets to zero after spike)
            # Vm = Vm * (1 - spikes.detach()) 
            #spike_prev = spikes
            out_seq.append(spikes)
            #vm_seq.append(Vm) # for FCReadout

        out_seq = torch.stack(out_seq, dim=1)
        #vm_seq = torch.stack(vm_seq, dim=1)
        #return out_seq, (Vm, out_seq[:, -1, :], adapt), vm_seq   # for FC Readout
        return out_seq, (Vm, out_seq[:, -1, :], adapt) # Normal without recurrence
        #return out_seq, (Vm, spike_prev, adapt)
    def init_state(self, batch_size, device=None):
        return (
            torch.zeros(batch_size, self.num_units, device=device),
            torch.zeros(batch_size, self.num_units, device=device),
            torch.zeros(batch_size, self.num_units, device=device),
        )

    