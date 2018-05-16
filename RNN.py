
import torch
from torch import nn
from torch.autograd import Variable


class BiRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Set initial states
        # print(x.data)
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN
        # 24*7*4
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bn = nn.BatchNorm2d(input_size, momentum=0.8)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        # print(x.data)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN
        # 24*7*4
        x = self.bn(x)
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

