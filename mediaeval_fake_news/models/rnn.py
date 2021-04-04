import torch as th
import torch.nn as nn


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Rnn, self).__init__()

        self.hidden_size = hidden_size

        self.rec_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='relu', batch_first=True)


    def forward(self, input):
        rec_out, _ = self.rec_layer(input)
        print(rec_out.shape)
        return rec_out

    def initHidden(self):
        return th.zeros(1, self.hidden_size)