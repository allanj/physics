import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F

class ContLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, sent_len, cuda):
        super(ContLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = cuda
        self.sent_len = sent_len
        self.input_dim = input_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = self.input_dim,
                            hidden_size = hidden_dim // 2,
                            bias = True,
                            bidirectional = True,
                            num_layers = 1,
                            batch_first = True
                            )
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, 1)

    # def init_hidden(self):
    #     h0_tensor = torch.zero(2, self.batch_size, self.hidden_dim // 2)
    #     c0_tensor = torch.zero(2, self.batch_size, self.hidden_dim // 2)
    #     if self.use_gpu:
    #         h0_tensor = h0_tensor.cuda()
    #         c0_tensor = c0_tensor.cuda()
    #     return (autograd.Variable(h0_tensor),
    #             autograd.Variable(c0_tensor))

    def forward(self, sentences):
        lstm_out, self.hidden = self.lstm(sentences)
        score = self.hidden2tag(lstm_out)
        last_score = F.sigmoid(score)
        return last_score