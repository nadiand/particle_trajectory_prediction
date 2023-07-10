import torch.nn as nn
import torch


class RegressionModel(nn.Module):
    '''
    A feedforward network for regression, which takes the hits (i.e 2D
    x and y coordinates) and outputs the corresponding trajectories.
    '''
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(RegressionModel, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size*2)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size*2, hidden_size*4)
        self.hidden3 = nn.Linear(hidden_size*4, hidden_size*2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden4 = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def forward(self, input):
        x = self.input_layer(input)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.hidden4(x)
        output = self.output_layer(x)
        output = torch.mean(output, dim=1)
        return output

    def init_weights(self, init_range=0.1) -> None:
        self.input_layer.weight.data.uniform_(-init_range, init_range)
        self.input_layer.bias.data.zero_()
        self.hidden1.weight.data.uniform_(-init_range, init_range)
        self.hidden1.bias.data.zero_()
        self.hidden2.weight.data.uniform_(-init_range, init_range)
        self.hidden2.bias.data.zero_()
        self.hidden3.weight.data.uniform_(-init_range, init_range)
        self.hidden3.bias.data.zero_()
        self.hidden4.weight.data.uniform_(-init_range, init_range)
        self.hidden4.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-init_range, init_range)
        self.output_layer.bias.data.zero_()
