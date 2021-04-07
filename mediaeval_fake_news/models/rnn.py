import torch as th
import torch.nn as nn
from torch.nn.modules import dropout


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Rnn, self).__init__()

        self.hidden_size = hidden_size

        self.rec_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh', batch_first=True)
        self.linear1 = nn.Linear(hidden_size, output_size)


    def forward(self, input):
        rec_out, _ = self.rec_layer(input)
        last_out = rec_out[0, -1, :]
        out = self.linear1(last_out)
        return out

    def initHidden(self):
        return th.zeros(1, self.hidden_size)