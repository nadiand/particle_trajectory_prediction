import torch.nn as nn
import torch

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(RegressionModel, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size*2)
        self.hidden2 = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weights()

    def forward(self, input):
        x = self.input_layer(input)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        output = torch.mean(output, dim=1)
        output = self.softmax(output)
        return output


    def init_weights(self, init_range=0.1) -> None:
        self.input_layer.weight.data.uniform_(-init_range, init_range)
        self.input_layer.bias.data.zero_()
        self.hidden1.weight.data.uniform_(-init_range, init_range)
        self.hidden1.bias.data.zero_()
        self.hidden2.weight.data.uniform_(-init_range, init_range)
        self.hidden2.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-init_range, init_range)
        self.output_layer.bias.data.zero_()

# should work for 2d data only, with 3 tracks
# input should be the xy and output should be the trajectories
# so input size might be (32x15x2) and output (32x3)