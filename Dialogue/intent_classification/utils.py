# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: utils.py
@time: 19-11-12 下午9:03

"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import json

def load_pre_embedding(data_path, word2id, embed_size):

    word_embed = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2id), embed_size))
    count = 0
    with open(data_path,  errors='ignore', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split(' ')
            if len(line) <= 2:
                continue
            if line[0] in word2id:
                word_embed[word2id[line[0]]] = np.array([float(i) for i in line[1:]])
                count += 1
    print('load pre word embedding {} / {} = {:.4f}'.format(count, len(word2id), count/len(word2id)))
    return word_embed


def evaluate(model, datas, use_gpu):
    true_num = 0.
    total_num = 0.
    errors = []
    for data in datas:
        text_ids = data['text_ids']
        # text = data['text']
        text_ids = Variable(torch.LongTensor(text_ids))
        label = data['label']
        if use_gpu:
            socre = model(text_ids.cuda())
        else:
            socre = model(text_ids)

        socre = torch.nn.Softmax()(socre)

        pred = torch.argmax(socre).item()
        #print('eval', el_targets, el_socre)
        total_num += 1
        if label == pred:
            true_num += 1
        else:
            data['pred'] = pred
            errors.append(data)
    with open('./data/errors.txt', 'w', encoding='utf-8') as f:
        for line in errors:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    acc = true_num / total_num

    return acc



def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


class EarlyStopping():
    def __init__(self, delta=0, patience=30):
        self.delta = delta
        self.patience = patience

        self.num_bad = 0
        self.current_loss = None

    def __call__(self, loss):

        if torch.isnan(loss):
            print('loss is NAN!')
            return True

        if self.current_loss is None:
            self.current_loss = loss
            return False

        if self.current_loss - self.delta > loss:
            self.current_loss = loss
            self.num_bad = 0
        else:
            self.num_bad += 1

        if self.num_bad >= self.patience:
            return True

        return False
