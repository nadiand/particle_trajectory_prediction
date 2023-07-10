"""
Transformer implementation. Built on the implementation of Saul A. M.:
https://github.com/saulam/trajectory_fitting/tree/main
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from global_constants import DIM


class EarthMoverLoss(nn.Module):
    def __init__(self):
        super(EarthMoverLoss, self).__init__()

    def forward(self, predicted_distribution, target_distribution):
        distance = torch.square(torch.cumsum(target_distribution, dim=-1) - torch.cumsum(predicted_distribution, dim=-1))
        return torch.mean(torch.mean(distance, dim=tuple(range(1, distance.ndim))))


class TransformerModel(nn.Module):
    '''
    A transformer network for trajectory reconstruction, which takes the hits 
    (i.e 2D or 3D coordinates) and outputs the corresponding trajectory parameter(s).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.decoder1 = nn.Linear(d_model, output_size)
        self.decoder2 = nn.Linear(d_model, output_size)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-init_range, init_range)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-init_range, init_range)
        if DIM == 3:
            self.decoder2.bias.data.zero_()
            self.decoder2.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, mask, padding_mask):
        src_emb = self.input_layer(input)
        memory = self.encoder(src=src_emb, mask=mask, src_key_padding_mask=padding_mask)
        memory = torch.mean(memory, dim=0)
        memory = self.dropout(memory)

        # Predicting 1 or 2 angles, depending on dimensionality
        if DIM == 2:
            out = self.decoder1(memory)
            return out
        else:
            return self.decoder1(memory), self.decoder2(memory)