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
from preprocess import prepare_ner_data
from utils import load_pre_embedding, evaluate_ner, evaluate_el
from er_model import BiLSTM_CRF
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
        # if use_gpu:
        #     score = er_model(text_ids.cuda())
        # else:
        #     score = er_model(text_ids)
        # _, out = torch.max(score, dim=-1)
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


    er_model = BiLSTM_CRF(len(word2id), config.embed_size, config.hidden_size, tag2id, None, config.dropout, use_gpu)
    if os.path.exists(config.model_path + '/ner_model/model.ckpt'):
        er_model.load_state_dict(torch.load(config.model_path + '/ner_model/model.ckpt'))
        print('load er_model state dict successful!')
    if use_gpu:
        er_model = er_model.cuda()
    er_model.eval()
    # dev
    print('=====dev=====')

    dev_er_p, dev_er_r, dev_er_f, dev_entity_p, dev_entity_r, dev_entity_f, dev_res_entitys, dev_res_positions = eval_ner(er_model, dev_data, id2tag, use_gpu)
    print("ER BIO: dev_p:{:.4f}, dev_r:{:.4f}, dev_F:{:.4f}".format(dev_er_p, dev_er_r, dev_er_f))
    print("ER Entity: true:{:.4f}, pred:{:.4f}, glod:{:.4f}".format(dev_entity_p, dev_entity_r, dev_entity_f))
    


    # train
    print('=====train=====')

    train_er_p, train_er_r, train_er_f, train_entity_p, train_entity_r, train_entity_f, train_res_entitys, train_res_positions = eval_ner(er_model, train_data, id2tag, use_gpu)
    print("ER BIO: train_p:{:.4f}, train_r:{:.4f}, train_F:{:.4f}".format(train_er_p, train_er_r, train_er_f))
    print("ER Entity: true:{:.4f}, pred:{:.4f}, glod:{:.4f}".format(train_entity_p, train_entity_r, train_entity_f))



