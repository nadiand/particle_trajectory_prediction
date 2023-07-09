from torch import Tensor
import torch
import torch.nn as nn
from itertools import combinations
import numpy as np

from sklearn import metrics
from global_constants import MAX_NR_TRACKS

# def pairs(targets, class_lbl):

# points in the same class i nthe reference should be in the same class in the prediction too
class ClusteringLoss(nn.Module):
    def __init__(self):
        super(ClusteringLoss, self).__init__()

    def forward(self, predicted_distribution, target_distribution):
        # print(predicted_distribution)
        # print(target_distribution)
        loss = 0.
        comparisons = 0
        for ind in range(len(target_distribution)):
            targets = target_distribution[ind]
            for class_lbl in range(MAX_NR_TRACKS):
                points_class_i = []
                for i, t in enumerate(targets):
                    if t[class_lbl] == 1:
                        points_class_i.append(i)
                pairs = list(combinations(points_class_i, 2))
                for ind1,ind2 in pairs:
                    comparisons += 1
                    if predicted_distribution[0][ind1].argmax() == predicted_distribution[0][ind2].argmax():
                        loss += 1
                    else: 
                        loss -= 1
        return torch.tensor(loss/comparisons, requires_grad=True)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1,  # dropout value
                 ):
        super(TransformerClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        # add another linear layer, try which location is good
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax() #i think dim=1 TODO test it out
        self.init_weights()


    def init_weights(self, init_range=0.1) -> None:
        # Initialization of weights and biases
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)


    def forward(self, input: Tensor, mask: Tensor, padding_mask: Tensor):
        # Linear layer
        src_emb = self.input_layer(input)
        # Transformer
        memory = self.encoder(src=src_emb, mask=mask, src_key_padding_mask=padding_mask)
        # Removing some weights for higher generalization
        memory = self.dropout(memory)
        # Linear layer
        out = self.decoder(memory)
        # Ensuring there are no negative probabilities
        out = self.softmax(out)
        # print(out.shape)
        # print(out)
        return out