"""
Transformer implementation.
Taken from https://github.com/saulam/trajectory_fitting/tree/main
"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from global_constants import DIM 

class EarthMoverLoss(nn.Module):
    def __init__(self):
        super(EarthMoverLoss, self).__init__()

    def forward(self, predicted_distribution, target_distribution):
        distance = torch.square(torch.cumsum(target_distribution, dim=-1) - torch.cumsum(predicted_distribution, dim=-1))
        return torch.mean(torch.mean(distance, dim=tuple(range(1, distance.ndim))))

    # def forward(self, predicted_distribution, target_distribution):
    #     y_pred = predicted_distribution / (torch.sum(predicted_distribution, dim=-1, keepdim=True) + 1e-14)
    #     y_true = target_distribution / (torch.sum(target_distribution, dim=-1, keepdim=True) + 1e-14)
    #     # print(y_pred)
    #     cdf_ytrue = torch.cumsum(y_true, axis=-1)
    #     cdf_ypred = torch.cumsum(y_pred, axis=-1)
    #     # print(cdf_ypred)
    #     samplewise_emd = torch.sqrt(torch.mean(torch.square(torch.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    #     return torch.mean(samplewise_emd)

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
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.decoder1 = nn.Linear(d_model, output_size)
        self.decoder2 = nn.Linear(d_model, output_size)
        self.init_weights()


    def init_weights(self, init_range=0.1) -> None:
        # Initialisation of weights and biases
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-init_range, init_range)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-init_range, init_range)
        if DIM == 3:
            self.decoder2.bias.data.zero_()
            self.decoder2.weight.data.uniform_(-init_range, init_range)


    def forward(self, input: Tensor, mask: Tensor, padding_mask: Tensor):
        # Linear layer
        src_emb = self.input_layer(input)
        # Transformer
        memory = self.encoder(src=src_emb, mask=mask, src_key_padding_mask=padding_mask)
        # Reducing a dimension TODO why
        memory = torch.mean(memory, dim=0)
        # Removing some weights for higher generalization
        memory = self.dropout(memory)

        # Predicting 1 or 2 angles, depending on dimensionality
        if DIM == 2:
            out = self.decoder1(memory)
            return out
        else:
            return self.decoder1(memory), self.decoder2(memory)