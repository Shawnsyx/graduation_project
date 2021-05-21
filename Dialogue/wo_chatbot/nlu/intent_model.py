#!/usr/bin/python
# -*- coding:utf8 -*-


import torch
import torch.nn as nn

class Intent_Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pre_word_embed=None, dropout=0.5, use_gpu=False):
        super(Intent_Model, self).__init__()
        self.use_gpu = use_gpu
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.word_embed = nn.Embedding(vocab_size, embed_size)

        if pre_word_embed is not None:
            self.word_embed.weight = nn.Parameter(torch.FloatTensor(pre_word_embed))
            self.pre_word_embed = True
        else:
            self.pre_word_embed = False


        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(hidden_size * 2, 13)


    def forward(self, sentence):
        lstm_out = self.get_lstm_features(sentence)
        score = self.output_layer(lstm_out)
        return score

    def get_lstm_features(self, sentence):
        embed = self.word_embed(sentence)
        embed = embed.unsqueeze(0)
        embed = self.dropout(embed)
        _, (lstm_out, _) = self.lstm(embed)
        lstm_out = lstm_out.view(-1, self.hidden_size * 2)
        lstm_out = self.dropout(lstm_out)
        return lstm_out
