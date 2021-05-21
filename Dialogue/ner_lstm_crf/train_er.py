# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: train.py
@time: 19-11-12 下午9:03

"""
import os
import time
import pickle
import torch
import numpy as np
from torch.autograd import Variable
from config import make_config
from preprocess import prepare_ner_data
from utils import load_pre_embedding, evaluate_ner
from er_model import BiLSTM_CRF, BiLSTM

t_start = time.time()
config = make_config()

with open(config.train_data_path, 'rb') as f:
    train_data = pickle.load(f)
with open(config.dev_data_path, 'rb') as f:
    dev_data = pickle.load(f)
# with open(config.test_data_path, 'rb') as f:
#     test_data = pickle.load(f)
with open(config.word_tag_path, 'rb') as f:
    word_tag = pickle.load(f)


tag2id = word_tag['tag2id']
id2tag = word_tag['id2tag']
word2id = word_tag['word2id']
id2word = word_tag['id2word']

prepare_ner_data(train_data, word2id, tag2id)
prepare_ner_data(dev_data, word2id, tag2id)
# prepare_ner_data(test_data, word2id, tag2id)

print('%i/%i/%i sentences in train/dev/test data'%(len(train_data), len(dev_data), len(test_data)))

use_gpu = False
if config.er_gpu is not None and torch.cuda.is_available():
    torch.cuda.set_device(int(config.er_gpu))
    use_gpu = True

pre_embed = None
# if config.use_pre_embedding:
#     pre_embed = load_pre_embedding('../pre_embedding/sgns.baidubaike.bigram-char', word2id, config.embed_size)

model = BiLSTM_CRF(len(word2id), config.embed_size, config.hidden_size, tag2id, pre_embed, config.dropout, use_gpu)

# if os.path.exists(config.model_path+'/ner_model/model.ckpt'):
#     model.load_state_dict(torch.load(config.model_path+'/ner_model/model.ckpt'))
#     print('load model state dict successful!')

if use_gpu:
    model = model.cuda()

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
best_dev_F1 = -1.0
loss = 0.0
b_loss = 0.
final_train_F = -1.0
time1 = time.time()
for epoch in range(config.er_epochs):
    np.random.shuffle(train_data)
    for i, data in enumerate(train_data):
        text_ids = data['text_ids']
        tag_ids = data['tag_ids']

        text_ids = Variable(torch.LongTensor(text_ids))
        tag_ids = torch.LongTensor(tag_ids)

        # if use_gpu:
        #     score = model(text_ids.cuda())
        #     tag_ids = tag_ids.cuda()
        # else:
        #     score = model(text_ids)
        # l = loss_function(score, tag_ids)
        # loss += l.data
        # b_loss += l

        # lstm + crf
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(text_ids.cuda(), tag_ids.cuda())
        else:
            neg_log_likelihood = model.neg_log_likelihood(text_ids, tag_ids)
        loss += neg_log_likelihood.data / len(text_ids)
        b_loss += neg_log_likelihood


        # model.zero_grad()
        # neg_log_likelihood.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        # optimizer.step()
        #print(loss)
        if (i+1) % config.batch_size == 0:
            model.zero_grad()
            b_loss /= config.batch_size
            b_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()
            b_loss = 0.

        if (i + 1) % 100 == 0:
            print("Epoch: {}/{}, step: {}/{}, Loss: {}, Time:{}".format(epoch + 1, config.er_epochs, i + 1, len(train_data), loss, time.time()- time1))
            loss = 0.
            time1 = time.time()

        if (i + 1) % 2000 == 0:#and epoch > 10:
            model.train(False)

            dev_p, dev_r, new_dev_F, entity_p, entity_r, entity_f = evaluate_ner(model, dev_data, config, id2tag, use_gpu)

            # test_p, test_r, test_F, test_entity_p, test_entity_r, test_entity_f = evaluate_ner(model, test_data, config, id2tag, use_gpu)

            if new_dev_F > best_dev_F1:
                best_dev_F1 = new_dev_F
                # final_test_F = test_F
                final_dev_entity = entity_f
                # final_test_entity = test_entity_f
                torch.save(model.state_dict(), config.model_path+'/model/model.ckpt')

            model.train(True)

            print("ER BIO:dev_p:{:.4f}, dev_r:{:.4f}, dev_F:{:.4f}".format(dev_p, dev_r, new_dev_F))
            print("ER Entity:dev_p:{:.4f}, dev_r:{:.4f}, dev_F:{:.4f}".format(entity_p, entity_r, entity_f))

            # print("ER BIO:test_p:{:.4f}, test_r:{:.4f}, test_F:{:.4f}".format(test_p, test_r, test_F))
            # print("ER Entity:test_p:{:.4f}, test_r:{:.4f}, test_F:{:.4f}".format(test_entity_p, test_entity_r, test_entity_f))

t_end = time.time()
# print('best dev F:{:.4f}, final test f:{:.4f}'.format(best_dev_F1, final_test_F))
# print('best dev entity F:{:.4f}, final test entity f:{:.4f}'.format(final_dev_entity, final_test_entity))
print('Train end! cost time:{}'.format(t_end - t_start))
