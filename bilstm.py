import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F

class SimpleLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size, sent_len, cuda):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = cuda
        self.sent_len = sent_len
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_dim // 2,
                            bias = True,
                            bidirectional = True,
                            num_layers = 1,
                            batch_first = True
                            )
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def init_hidden(self):
        h0_tensor = torch.randn(2, self.batch_size, self.hidden_dim // 2)
        c0_tensor = torch.randn(2, self.batch_size, self.hidden_dim // 2)
        if self.use_gpu:
            h0_tensor = h0_tensor.cuda()
            c0_tensor = c0_tensor.cuda()
        return (autograd.Variable(h0_tensor),
                autograd.Variable(c0_tensor))

    def forward(self, sentences):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentences)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores