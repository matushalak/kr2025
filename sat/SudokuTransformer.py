# Author: @matushalak
""""
Full definition of SudokuTransformer in this file

Some functions are taken / adapted from Assignment 2 of DL1 course at UvA
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple

class AttConfig:
    def __init__(self, n_embd=12, n_head=3, attn_pdrop=0.01, resid_pdrop=0.05,
                 recurrence = 1):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.recurrence = recurrence

class SudokuTransformer(nn.Module):
    def __init__(self, config:AttConfig = AttConfig(), 
                 sudoku_size:int = 9, 
                 n_transformer_blocks:int = 5,
                 mlp_expansion:int = 4):
        super().__init__()
        self.config = config
        if self.config.recurrence < 1:
            self.config.recurrence = 1
        self.sudoku_size = sudoku_size

        # Each 0-9 digit gets its embedding
        self.DigitEmbedding = nn.Embedding(sudoku_size+1, embedding_dim=self.config.n_embd)
        # Each 1-9 row gets its embedding
        self.RowEmbedding = nn.Embedding(sudoku_size, embedding_dim=self.config.n_embd)
        # Each 1-9 col gets its embedding
        self.ColEmbedding = nn.Embedding(sudoku_size, embedding_dim=self.config.n_embd)
        # Each 1-9 box gets its embedding
        self.BoxEmbedding = nn.Embedding(sudoku_size, embedding_dim=self.config.n_embd)
        
        # Pre-computing structure of problem (re-use each forward pass)
        rows, colx, boxes = self.sudoku_positions()
        self.register_buffer('rows', rows)
        self.register_buffer('cols', rows)
        self.register_buffer('boxes', rows)

        # Chain of Transformer blocks
        self.Transformer = nn.Sequential(*[TransformerBlock(self.config, 
                                                            mlp_hidden=mlp_expansion*self.config.n_embd) 
                                           for b in range(n_transformer_blocks)])
        self.Logits = nn.Linear(self.config.n_embd, sudoku_size)

    def sudoku_positions(self):
        N = self.sudoku_size
        sizesqrt = int(N ** 0.5)
        T = N * N
        idx = torch.arange(T) # (T,)
        rows = idx // N # (T,)
        cols = idx % N # (T,)
        boxes = (rows // sizesqrt) * sizesqrt + (cols // sizesqrt) # (T,)
        return rows, cols, boxes

    def forward(self, input_batch_digits):
        # Input (B, T)
        B, T = input_batch_digits.shape
        # Absolute Sudoku embeddings to inject inductive bias about board structure
        # Broadcast over batch
        input_batch_rows = self.rows.unsqueeze(0).expand(B, -1)   # (B, T)
        input_batch_cols = self.cols.unsqueeze(0).expand(B, -1)   # (B, T)
        input_batch_boxes = self.boxes.unsqueeze(0).expand(B, -1) # (B, T)
        # Get all absolute embeddings for each entry
        de = self.DigitEmbedding(input_batch_digits)
        re = self.RowEmbedding(input_batch_rows)
        ce = self.ColEmbedding(input_batch_cols)
        be = self.BoxEmbedding(input_batch_boxes)
        x = de + re + ce + be
        # Run through a chain of transformer blocks
        # XXX: recurrence test -> significant performance boost
        for _ in range(self.config.recurrence):
            x = self.Transformer(x)
        # Run through final linear layer
        # (B, T, d_embd) @ (d_embd, sudoku_size) -> (B, T, sudoku_size)
        # Applied separately for each token over the embedding dimension
        x = self.Logits(x)
        # For each token, produce logits for which digit is predicted there
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config, mlp_hidden:int):
        super().__init__()
        self.LayerNormMHA = nn.LayerNorm(normalized_shape=config.n_embd)
        self.MHA = SelfAttention(config)
        self.LayerNormMLP = nn.LayerNorm(normalized_shape=config.n_embd)
        self.MLP = MLP(d_embd = config.n_embd, d_expand=mlp_hidden)
    
    def forward(self, x):
        # Residual connection + MHA -> y
        y = self.LayerNormMHA(x + self.MHA(x))
        # Residual connection + MLP -> out
        out = self.LayerNormMLP(y + self.MLP(y))
        return out

class MLP(nn.Module):
    def __init__(self, d_embd:int, d_expand:int):
        super().__init__()
        self.L1 = nn.Linear(d_embd, d_expand)
        self.L2 = nn.Linear(d_expand, d_embd)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # First layer - expand + nonlinear
        x = self.L1(x) # expand
        x = self.gelu(x) # nonlinear
        # Second layer - linear contract
        x = self.L2(x)
        return x 
    
class SelfAttention(nn.Module):
    '''
    No Causal mask, want ALL-to-ALL attention
    '''
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 and config.n_embd % 2 == 0
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
        seq_pos = torch.arange(T, device = xq.device)  # Shape: (T)
        # for rotary positional embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2, device = xq.device).float() / self.d_k))
        pos_angles = seq_pos[:, None] * inv_freq[None, :] # Shape: (T, dim // 2)
        
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
    
    def forward(self, x):
        B, T, d_embd = x.shape
        assert d_embd == self.d_embd
        # Linear projection to all heads at once
        allheads = self.W_embed(x) # (B, T, 3*d_embd)
        # add head dimension -> (B, nh, T, 3*d_k)
        allheads = allheads.view(B, T, 3*self.d_k, self.n_head) 
        allheads = torch.permute(allheads, (0, 3, 1, 2))

        # Split into Q, K, V each with shape (B, nh, T, d_k)
        q, k ,v  = torch.tensor_split(allheads, 3, dim = -1)
        
        # Apply positional embeddings
        q, k = self.apply_rotary_emb(q, k, T)
        
        # Compute attention scores
        # (similarity) Scaled Dot Product = (Q@K.T)/sqrt(d_k)
        similarity = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)
        
        # Apply Softmax (masked entries with -inf effectively zeroed out)
        att = torch.softmax(similarity, dim = -1)

        # Apply dropout on attention matrix
        att = self.attn_dropout(att)
        
        # Apply attention to the values
        # (B, nh, T, T) x (B, nh, T, d_v) -> (B, nh, T, d_v)
        y = torch.matmul(att, v)
        # re-assemble (concatenate) all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, d_embd) # (B, T, nh * d_v)
        # Pass through final linear layer: (B, T, nh * d_v) x (1, 1, nh * d_v, d_embd)
        y = self.W_out(y) # (B, T, nh * d_embd)
        # Apply dropout on MHA output
        y = self.resid_dropout(y)
        return y