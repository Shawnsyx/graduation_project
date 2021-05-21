# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: ed_model.py
@time: 19-11-22 下午1:44

"""
import torch
import torch.nn as nn
from utils import init_embedding, init_linear, init_lstm

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
            init_embedding(self.word_embed.weight)

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        init_lstm(self.lstm)
        self.output_layer = nn.Linear(hidden_size * 2, 13)
        init_linear(self.output_layer)

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
