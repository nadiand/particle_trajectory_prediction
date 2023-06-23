"""
Transformer implementation.
Taken from https://github.com/saulam/trajectory_fitting/tree/main
"""
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FittingTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1,  # dropout value
                 seq_len: int = 15):
        super(FittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.aggregator = nn.Linear(seq_len, 1)
        self.decoder = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        # weights initialisation
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self,
                x: Tensor,  # Separate x tensor
                y: Tensor,  # Separate y tensor
                z: Optional[Tensor] = None,  # Separate z tensor; works with None
                ):
        # Initialize a tensor of zeros with the same size as the x and y tensors when z is None
        if z is None:
            z = torch.zeros_like(x)

        # Combine x, y, and z tensors into coords tensor of shape (batch_size, sequence_length, 3)
        coords = torch.stack((x, y, z), dim=2)

        # Linear projection of the input
        src_emb = self.proj_input(coords)
        # Transformer encoder
        memory = self.transformer_encoder(src=src_emb)
        memory = self.aggregator(memory.permute(0, 2, 1))
        memory = memory.squeeze(dim=2)
        # Dropout
        memory = self.dropout(memory)
        # Linear projection of the output
        output = self.decoder(memory) #+ coords[:, :, :3]  # Learn residuals for x, y, z

        return output