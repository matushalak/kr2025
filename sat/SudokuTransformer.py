#Author: @matushalak
""""
Full definition of SudokuTransformer in this file

Some functions are taken / adapted from Assignment 2 of DL1 course at UvA
"""
import torch
import numpy as np

import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from typing import Tuple

class AttConfig:
    def __init__(self, n_embd=10, n_head=8, attn_pdrop=0.05, resid_pdrop=0.1, block_size=81):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.block_size = block_size

class SudokuTransformer:
    def __init__(self):
        pass

class TransformerEncoder:
    def __init__(self):
        pass

    
class SudokuAttention:
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # important dimensions
        self.n_head = config.n_head
        self.d_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head
        # key, query, value projections for all heads, but in a batch
        self.W_embed = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.W_out = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embeddings using sine and cosine functions to the query and key tensors.
        
        Args:
            xq (torch.Tensor): Query tensor of shape [batch, num_heads, seq_len, head_dim].
            xk (torch.Tensor): Key tensor of shape [batch, num_heads, seq_len, head_dim].
            T int: sequence length
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the modified query and key tensors.
        """
        b, h, t, d = xq.shape
        # Generate RoPE embeddings dynamically based on T
        seq_pos = torch.arange(T)  # Shape: (T)
        pos_angles = seq_pos[:, None] * self.inv_freq[None, :]    # Shape: (T, dim // 2)
        
        # Split pos into sin and cos components, repeating each to match xq and xk dimensions
        # (1, 1, T, dim // 2)
        pos_sin = torch.sin(pos_angles)[None, None, ...]
        pos_cos = torch.cos(pos_angles)[None, None, ...]

        # Apply RoPE transformation: pair and rotate dimensions
        # Rotate query and key tensors (split between even and odd embedding dimensions)
        # apply to queries
        xq_rot_even = pos_cos * xq[..., 0::2] - pos_sin * xq[..., 1::2]
        xq_rot_odd = pos_sin * xq[..., 0::2] + pos_cos * xq[..., 1::2]
        xq_rot = torch.stack((xq_rot_even, xq_rot_odd), dim = - 1).view(b,h,t,d)
        # apply to keys
        xk_rot_even = pos_cos * xk[..., 0::2] - pos_sin * xk[..., 1::2]
        xk_rot_odd = pos_sin * xk[..., 0::2] + pos_cos * xk[..., 1::2]
        xk_rot = torch.stack((xk_rot_even, xk_rot_odd), dim = - 1).view(b,h,t,d)
        
        return xq_rot, xk_rot
    
    def make_mask(self):
        pass

    def forward(self, x):
        B, T, d_embd = x.shape
        assert d_embd == self.d_embd
        # Linear projection to all heads at once
        allheads = self.W_embed(x) # (B, T, 3*d_embd)
        # add head dimension -> (B, nh, T, 3*d_k)
        allheads = allheads.view(B, T, d_embd, self.n_head) 
        allheads = torch.permute(allheads, (0, 3, 1, 2))
        # Split into Q, K, V each with shape (B, nh, T, d_k)
        q, k ,v  = torch.tensor_split(allheads, 3, dim = -1)
        # Apply positional embeddings
        q, k = self.apply_rotary_emb(q, k, T)
        y = self.resid_dropout(self.W_out(y))
        return y