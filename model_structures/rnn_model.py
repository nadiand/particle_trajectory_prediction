import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from global_constants import *


class RNNModel(nn.Module):
    '''
    A RNN network meant for trajectory parameter regression. It takes a group
    of hits (i.e 2D or 3D coordinates) and outputs the trajectory parameter(s)
    associated with that group. In the case of 3D hits, there are two output
    layers, each predicting one separate parameter.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out1 = nn.Linear(hidden_size, output_size)
        self.out2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.init_weights()

    def forward(self, input, input_lens):
        batch = input.shape[0]
        h_0 = torch.zeros(1, batch, self.hidden_size).to(input.device)
        c_0 = torch.zeros(1, batch, self.hidden_size).to(input.device)
        packed_input = pack_padded_sequence(input, input_lens, batch_first=True, enforce_sorted=False)
        _, (final_hidden_state, _) = self.lstm(packed_input, (h_0, c_0))

        out = self.out1(final_hidden_state[-1])
        out = out.squeeze(dim=1)
        if DIM == 3:
            out2 = self.out2(final_hidden_state[-1]) 
            out2 = out2.squeeze(dim=1)
            return out, out2
        return out
    
    def init_weights(self, init_range=0.1) -> None:
        self.out1.bias.data.zero_()
        self.out1.weight.data.uniform_(-init_range, init_range)
        if DIM == 3:
            self.out2.bias.data.zero_()
            self.out2.weight.data.uniform_(-init_range, init_range)