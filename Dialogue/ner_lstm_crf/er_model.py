# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: ner_model.py
@time: 19-11-12 下午9:03

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import init_embedding, init_linear, init_lstm
import torch.nn.functional as F
class CNN_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, filter_sizes=[3, 5]):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.filter_sizes = filter_sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, hidden_size, (filter_size, input_size), padding=(filter_size//2, 0)) for filter_size in filter_sizes
            ]
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        outputs = [conv(inputs) for conv in self.convs]
        outputs = [output.transpose(1, 3).squeeze(1) for output in outputs]
        outputs = outputs = torch.cat(outputs, dim=-1)
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, tag2id, pre_word_embed=None, dropout=0.5, use_gpu=False):
        super(BiLSTM, self).__init__()
        self.use_gpu = use_gpu
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.word_embed = nn.Embedding(vocab_size, embed_size)
        if pre_word_embed is not None:
            self.word_embed.weight = nn.Parameter(torch.FloatTensor(pre_word_embed))
            self.pre_word_embed = True
        else:
            self.pre_word_embed = False
            init_embedding(self.word_embed.weight)

        init_lstm(self.lstm)
        self.hidden2tag = nn.Linear(hidden_size * 2, self.tag_size)
        init_linear(self.hidden2tag)


    def forward(self, sentence):
        feats = self.get_lstm_features(sentence)
        score = F.log_softmax(feats, dim=-1)
        _, out = torch.max(score, dim=-1)
        return score

    def get_lstm_features(self, sentence):

        embed = self.word_embed(sentence)
        embed = embed.unsqueeze(0)

        embed = self.dropout(embed)
        lstm_out, _ = self.lstm(embed)
        lstm_out = lstm_out.view(len(sentence), self.hidden_size * 2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, tag2id, pre_word_embed=None, dropout=0.5, use_gpu=False):
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        # cnn after
        # self.cnn = CNN_Encoder(hidden_size, hidden_size)
        # self.bridge = nn.Linear(hidden_size*2, self.hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)

        # cnn before
        # self.cnn = CNN_Encoder(embed_size, hidden_size)
        # self.bridge = nn.Linear(hidden_size * 2, self.hidden_size)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        self.word_embed = nn.Embedding(vocab_size, embed_size)
        if pre_word_embed is not None:
            self.word_embed.weight = nn.Parameter(torch.FloatTensor(pre_word_embed))
            self.pre_word_embed = True
        else:
            self.pre_word_embed = False
            init_embedding(self.word_embed.weight)


        init_lstm(self.lstm)
        self.hidden2tag = nn.Linear(hidden_size*2, self.tag_size)
        init_linear(self.hidden2tag)
        self.tanh = nn.Tanh()

        # crf layer
        self.transitions = nn.Parameter(torch.zeros(self.tag_size, self.tag_size))
        self.transitions.data[tag2id['START'], :] = -10000
        self.transitions.data[:, tag2id['STOP']] = -10000

    def forward(self, sentence):
        feats = self.get_lstm_features(sentence)

        score, tag_seq = self.viterbi_decode(feats)
        return score, tag_seq


    def get_lstm_features(self, sentence):

        embed = self.word_embed(sentence)
        embed = embed.unsqueeze(1)
        lstm_out, _ = self.lstm(embed)
        lstm_out = lstm_out.view(len(sentence), self.hidden_size * 2)

        lstm_out = self.dropout(lstm_out)


        feats = self.hidden2tag(lstm_out)
        return feats

    def viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.Tensor(1, self.tag_size).fill_(-10000.)
        init_vvars[0][self.tag2id['START']] = 0
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tag_size, self.tag_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivar_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivar_t = Variable(torch.FloatTensor(viterbivar_t))
            if self.use_gpu:
                viterbivar_t = viterbivar_t.cuda()
            forward_var = viterbivar_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag2id['STOP']]
        terminal_var.data[self.tag2id['STOP']] = -10000.
        terminal_var.data[self.tag2id['START']] = -10000.
        _, best_tag_id = torch.max(terminal_var.unsqueeze(0), 1)
        best_tag_id = best_tag_id.view(-1).data.tolist()[0]
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2id['START']
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):

        feats = self.get_lstm_features(sentence)
        forward_score = self.forward_alg(feats)
        gold_score = self.score_sentence(feats, tags)
        if torch.isnan(forward_score):
            print(sentence, tags)
            print(feats)
        return forward_score - gold_score

    def score_sentence(self, feats, tags):

        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag2id['START']]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag2id['STOP']])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag2id['START']]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag2id['STOP']])])
        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])
        return score

    def forward_alg(self, feats):
        init_alphas = torch.Tensor(1, self.tag_size).fill_(-10000.)
        init_alphas[0][self.tag2id['START']] = 0.
        forward_var = torch.autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)
        terminal_val = (forward_var + self.transitions[self.tag2id['STOP']]).view(1, -1)
        alpha = log_sum_exp(terminal_val)

        return alpha

def log_sum_exp( vec):
    _, idx = torch.max(vec, 1)
    idx = idx.view(-1).data.tolist()[0]
    max_score = vec[0, idx]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

