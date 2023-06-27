"""
Transformer implementation.
Taken from https://github.com/saulam/trajectory_fitting/tree/main
"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import pairwise_distances
import numpy as np
import ot
from global_constants import DIM 


# class EarthMoverLoss(nn.Module):
#     def __init__(self):
#         super(EarthMoverLoss, self).__init__()

#     def forward(self, predicted_distribution, target_distribution):
#         distance_matrix = pairwise_distances(
#             predicted_distribution[:, np.newaxis],
#             target_distribution[:, np.newaxis],
#             metric='euclidean'
#         )
#         loss = ot.emd2(predicted_distribution, target_distribution, distance_matrix)
#         return torch.tensor(loss, requires_grad=True)


class TransformerModel(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1,  # dropout value
                 ):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.decoder1 = nn.Linear(d_model, output_size)
        self.decoder2 = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        # weights initialisation
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-init_range, init_range)
        if DIM == 2:
            self.decoder2.bias.data.zero_()
            self.decoder2.weight.data.uniform_(-init_range, init_range)

    def forward(self, input: Tensor, mask: Tensor, padding_mask: Tensor):
        # Linear projection of the input
        src_emb = self.proj_input(input)
        # Transformer encoder
        memory = self.transformer_encoder(src=src_emb, mask=mask, src_key_padding_mask=padding_mask)
        memory = torch.mean(memory, dim=0)
        memory = self.dropout(memory)
        # Linear projection of the output
        if DIM == 2:
            return self.decoder1(memory)
        else:
            return self.decoder1(memory), self.decoder2(memory)