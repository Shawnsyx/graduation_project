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

def evaluate_ner(model, datas, config, id2tag, use_gpu):
    pred = []
    new_F = 0.0

    true_num = 0
    pred_num = 0.0001
    glod_num = 0
    entity_true = 0
    entity_pred = 0.0001
    entity_glod = 0
    for data in datas:
        glod = data['tag_ids']
        text = data['text']
        text_ids = data['text_ids']

        text_ids = Variable(torch.LongTensor(text_ids))

        if use_gpu:
            val, out = model(text_ids.cuda())
        else:
            val, out = model(text_ids)
        # if use_gpu:
        #     score = model(text_ids.cuda())
        # else:
        #     score = model(text_ids)
        # _, out = torch.max(score, dim=-1)
        # out = out.data.cpu().numpy()
        for i, j in zip(out, glod):
            if id2tag[i] != 'O':
                pred_num += 1
            if id2tag[j] != 'O':
                glod_num += 1
            if i == j and id2tag[i] != 'O':
                true_num += 1
            if id2tag[i] == 'B':
                entity_pred += 1
            if id2tag[j] == 'B':
                entity_glod += 1

        _, positions = find_entity(text, out, id2tag)

        for i in positions:
            start, end = i
            flag = True
            for j in range(start, end + 1):
                if out[j] != glod[j]:
                    flag = False
                    break
            if end < len(out) - 1 and id2tag[glod[end + 1]] == 'I':
                flag = False
            if flag:
                entity_true += 1
        #print('eval!')

    p = true_num / pred_num
    r = true_num / glod_num
    if p + r == 0:
        new_F = 0
    else:
        new_F = 2 * p * r / (p + r)

    entity_p = entity_true / entity_pred
    entity_r = entity_true / entity_glod
    if entity_p + entity_r == 0:
        entity_f = 0
    else:
        entity_f = 2 * entity_p * entity_r / (entity_p + entity_r)

    return p, r, new_F, entity_p, entity_r, entity_f

def find_entity(text, out, id2tag):
    entity = []
    positions = []
    i = 0
    while i < len(out):
        if id2tag[out[i]] == 'B':
            start = i
            i += 1
            while i < len(out) and id2tag[out[i]] == 'I':
                i += 1
            i -= 1
            end = i
            positions.append([start, end])
            entity.append(text[start:end+1])
        i += 1
    return entity, positions

def evaluate_el(model, datas, use_gpu):
    true_num = 0
    pred_num = 0
    glod_num = 0

    for data in datas:
        text_ids = data['text_ids']
        # text = data['text']
        text_ids = Variable(torch.LongTensor(text_ids))
        mention_positions = data['mention_position']
        el_candidate = data['el_data']
        el_candidate = Variable(torch.LongTensor(el_candidate))
        el_targets = data['el_label']
        if use_gpu:
            el_socre = model(text_ids.cuda(), mention_positions, el_candidate.cuda())
        else:
            el_socre = model(text_ids, mention_positions, el_candidate)

        el_socre = torch.nn.Sigmoid()(el_socre).view(-1).tolist()[0]

        #print('eval', el_targets, el_socre)
        if el_socre > 0.5 and el_targets == 1:
            true_num += 1
        if el_socre > 0.5:
            pred_num += 1
        if el_targets == 1:
            glod_num += 1

    p = true_num / pred_num
    r = true_num / glod_num
    new_F = 2 * p * r / (p + r)

    return p, r, new_F, true_num, pred_num, glod_num



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
