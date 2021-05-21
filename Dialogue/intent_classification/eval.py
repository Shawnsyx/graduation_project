# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: eval.py
@time: 19-11-24 上午11:09

"""

import os
import time
import pickle
import torch
import numpy as np
from torch.autograd import Variable
from config import make_config
from preprocess import prepare_ner_data, prepare_ed_data
from utils import load_pre_embedding, evaluate_ner, evaluate_el
from er_model import BiLSTM_CRF
from ed_model import ED_Model
from utils import find_entity

def eval_ner(er_model, datas, id2tag, use_gpu):
    true_num = 0
    pred_num = 0
    glod_num = 0
    entity_true = 0
    entity_pred = 0
    entity_glod = 0
    res_entitys = []
    res_positions = []

    for data in datas:
        glod = data['tag_ids']
        text = data['text']
        text_ids = data['text_ids']

        text_ids = Variable(torch.LongTensor(text_ids))

        if use_gpu:
            val, out = er_model(text_ids.cuda())
        else:
            val, out = er_model(text_ids)
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

        entitys, positions = find_entity(text, out, id2tag)
        res_entitys.append(entitys)
        res_positions.append(positions)

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
        # print('eval!')

    p = true_num / pred_num
    r = true_num / glod_num
    new_F = 2 * p * r / (p + r)

    entity_p = entity_true / entity_pred
    entity_r = entity_true / entity_glod
    entity_f = 2 * entity_p * entity_r / (entity_p + entity_r)

    return p, r, new_F, entity_p, entity_r, entity_f, res_entitys, res_positions

def generate_candidate(mention, entity2id, id2desc, word2id):
    candidate_ids = entity2id[mention]
    candidate_desc = []
    if len(candidate_ids) == 0:
        return None, None
    for can in candidate_ids:
        desc = id2desc[can]
        candidate_desc.append([word2id[i] if i in word2id else word2id['<UNK>'] for i in desc])
    return candidate_ids, candidate_desc

def pipe_eval_el(ed_model, datas, res_entitys, res_positions, use_gpu, entity2id, id2desc, word2id):
    true_num = 0
    pred_num = 0
    glod_num = 0
    el_true = 0
    el_pred = 0
    el_glod = 0
    for index in range(len(datas)):
        glod = datas[index]['tag_ids']
        text_ids = datas[index]['text_ids']
        text = datas[index]['text']
        text_ids = Variable(torch.LongTensor(text_ids))
        mention_data = datas[index]['mention_data']
        res_entity = []
        res_kb = []
        res_pos = []
        entity = res_entitys[index]
        positions = res_positions[index]
        for i in range(len(entity)):
            candidate_ids, candidate_desc = generate_candidate(entity[i], entity2id, id2desc, word2id)
            #candidate_desc = Variable(torch.LongTensor(candidate_desc))
            if candidate_ids is None:
                continue
            if len(candidate_ids) == 1:
                res_entity.append(entity[i])
                res_kb.append(candidate_ids[0])
                res_pos.append(positions[i])
            else:
                mention_positions = [positions[i]] * len(candidate_ids)
                if use_gpu:
                    text_ids = text_ids.cuda()
                pred = -1
                pred_score = -1.0
                for j in range(len(candidate_ids)):
                    candidate_desc_j = Variable(torch.LongTensor(candidate_desc[j]))
                    if use_gpu:
                        candidate_desc_j = candidate_desc_j.cuda()
                    el_score = ed_model(text_ids, mention_positions[j], candidate_desc_j)
                    el_score = torch.nn.Sigmoid()(el_score).view(-1).tolist()[0]
                    if el_score > pred_score:
                        pred_score = el_score
                        pred = j
                res_entity.append(entity[i])
                res_kb.append(candidate_ids[pred])
                res_pos.append(positions[i])
        el_pred += len(res_entity)
        #print(text, mention_data)
        #print(entity, positions)
        #print(res_entity, res_kb, res_pos)
        for i in mention_data:
            mention = i['mention']
            start = int(i['offset'])
            kb_id = i['kb_id']
            if kb_id == 'NIL':
                continue
            el_glod += 1
            for index in range(len(res_entity)):
                if mention == res_entity[index] and kb_id == res_kb[index]:
                    el_true += 1
                    #print('123-',text, mention, kb_id, res_entity[index])
                    break
    #print(el_true, el_pred, el_glod)
    p = el_true / el_pred
    r = el_true / el_glod
    new_F = 2 * p * r / (p + r)
    return p, r, new_F

def eval_el_new(model, datas, use_gpu, word2id, id2entity, entity2id, id2desc):
    true_num = 0
    pred_num = 0
    glod_num = 0
    datas = prepare_ed_data(datas, word2id, id2entity, entity2id, id2desc)

    for data in datas:
        text_ids, mention_positions, el_candidate, el_targets = data['text_ids'], data['mention_position'], data['el_data'], data['el_label']
        # el_targets = torch.Tensor(el_targets).unsqueeze(1)
        text_ids = Variable(torch.LongTensor(text_ids))
        el_candidate = Variable(torch.LongTensor(el_candidate))

        if use_gpu:
            text_ids = text_ids.cuda()
            el_candidate = el_candidate.cuda()
        el_score = model(text_ids, mention_positions, el_candidate)
        el_score = torch.nn.Sigmoid()(el_score).view(-1).tolist()

        # print('eval', el_targets, el_socre)
        for i in range(len(el_targets)):
            if el_score[i] > 0.5 and el_targets[i] == 1:
                true_num += 1
            if el_score[i] > 0.5:
                pred_num += 1
            if el_targets[i] == 1:
                glod_num += 1
    if pred_num == 0:
        p = 0
    else:
        p = true_num / pred_num
    r = true_num / glod_num
    if p == 0 and r == 0:
        new_F = 0
    else:
        new_F = 2 * p * r / (p + r)

    return p, r, new_F, true_num, pred_num, glod_num
def eval_el(model, datas, use_gpu, word2id, id2entity, entity2id, id2desc):
    true_num = 0
    pred_num = 0
    glod_num = 0
    
    datas = prepare_ed_data(datas, word2id, id2entity, entity2id, id2desc)

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

if __name__ == '__main__':
    t_start = time.time()
    config = make_config()

    with open(config.train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(config.dev_data_path, 'rb') as f:
        dev_data = pickle.load(f)
    with open(config.word_tag_path, 'rb') as f:
        word_tag = pickle.load(f)
    with open(config.kb_path, 'rb') as f:
        kb_data = pickle.load(f)

    tag2id = word_tag['tag2id']
    id2tag = word_tag['id2tag']
    word2id = word_tag['word2id']
    id2word = word_tag['id2word']

    id2entity = kb_data['id2entity']
    entity2id = kb_data['entity2id']
    id2desc = kb_data['id2desc']
    
    #train_data = train_data[:10]
    #dev_data = dev_data[:10]
    prepare_ner_data(train_data, word2id, tag2id)
    prepare_ner_data(dev_data, word2id, tag2id)

    print('%i/%i sentences in train/dev data' % (len(train_data), len(dev_data)))

    use_gpu = False
    if config.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(config.gpu))
        use_gpu = True

    ed_model = ED_Model(len(word2id), config.embed_size, config.hidden_size, None, config.dropout, use_gpu)
    if os.path.exists(config.model_path + '/ed_model/model.ckpt'):
        ed_model.load_state_dict(torch.load(config.model_path + '/ed_model/model.ckpt'))
        print('load ed_model state dict successful!')

    er_model = BiLSTM_CRF(len(word2id), config.embed_size, config.hidden_size, tag2id, None, config.dropout, use_gpu)
    if os.path.exists(config.model_path + '/ner_model/model.ckpt'):
        er_model.load_state_dict(torch.load(config.model_path + '/ner_model/model.ckpt'))
        print('load er_model state dict successful!')
    if use_gpu:
        ed_model = ed_model.cuda()
        er_model = er_model.cuda()
    ed_model.eval()
    er_model.eval()
    # dev
    print('=====dev=====')

    dev_er_p, dev_er_r, dev_er_f, dev_entity_p, dev_entity_r, dev_entity_f, dev_res_entitys, dev_res_positions = eval_ner(er_model, dev_data, id2tag, use_gpu)
    print("ER BIO: dev_p:{:.4f}, dev_r:{:.4f}, dev_F:{:.4f}".format(dev_er_p, dev_er_r, dev_er_f))
    print("ER Entity: true:{:.4f}, pred:{:.4f}, glod:{:.4f}".format(dev_entity_p, dev_entity_r, dev_entity_f))
    
    dev_ed_p, dev_ed_r, dev_ed_f, _, _, _ = eval_el(ed_model, dev_data, use_gpu, word2id, id2entity, entity2id, id2desc)
    print("EL: dev_p:{:.4f}, dev_r:{:.4f}, dev_F:{:.4f}".format(dev_ed_p, dev_ed_r, dev_ed_f))

    pipe_dev_ed_p, pipe_dev_ed_r, pipe_dev_ed_f = pipe_eval_el(ed_model, dev_data, dev_res_entitys, dev_res_positions, use_gpu, entity2id, id2desc, word2id)
    print("Pipe-EL: dev_p:{:.4f}, dev_r:{:.4f}, dev_F:{:.4f}".format(pipe_dev_ed_p, pipe_dev_ed_r, pipe_dev_ed_f))


    # train
    print('=====train=====')

    train_er_p, train_er_r, train_er_f, train_entity_p, train_entity_r, train_entity_f, train_res_entitys, train_res_positions = eval_ner(er_model, train_data, id2tag, use_gpu)
    print("ER BIO: train_p:{:.4f}, train_r:{:.4f}, train_F:{:.4f}".format(train_er_p, train_er_r, train_er_f))
    print("ER Entity: true:{:.4f}, pred:{:.4f}, glod:{:.4f}".format(train_entity_p, train_entity_r, train_entity_f))

    train_ed_p, train_ed_r, train_ed_f,_, _, _ = eval_el(ed_model, train_data, use_gpu, word2id, id2entity, entity2id, id2desc)
    print("EL: train_p:{:.4f}, train_r:{:.4f}, train_F:{:.4f}".format(train_ed_p, train_ed_r, train_ed_f))

    pipe_train_ed_p, pipe_train_ed_r, pipe_train_ed_f = pipe_eval_el(ed_model, train_data, train_res_entitys, train_res_positions, use_gpu, entity2id, id2desc, word2id)
    print("Pipe-EL: train_p:{:.4f}, train_r:{:.4f}, train_F:{:.4f}".format(pipe_train_ed_p, pipe_train_ed_r, pipe_train_ed_f))




