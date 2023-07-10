import torch.nn as nn


class TransformerClassifier(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        # add another linear layer, try which location is good TODO
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=0)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, mask, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, mask=mask, src_key_padding_mask=padding_mask)
        memory = self.dropout(memory)
        out = self.decoder(memory)
        # Ensuring there are no negative probabilities
        out = self.softmax(out)
        return out