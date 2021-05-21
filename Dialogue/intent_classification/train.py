# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: train_ed.py
@time: 19-11-22 下午1:44

"""

import os
import time
import pickle
import torch
import numpy as np
from torch.autograd import Variable
from config import make_config
from utils import load_pre_embedding, EarlyStopping, evaluate
from model import Intent_Model

t_start = time.time()
config = make_config()

with open(config.train_data_path, 'rb') as f:
    train_data = pickle.load(f)
with open(config.dev_data_path, 'rb') as f:
    dev_data = pickle.load(f)
with open(config.word_tag_path, 'rb') as f:
    word_dic = pickle.load(f)



word2id = word_dic['word2id']
id2word = word_dic['id2word']



print('%i/%i sentences in train/dev data'%(len(train_data), len(dev_data)))

use_gpu = False
if config.ed_gpu is not None and torch.cuda.is_available():
    torch.cuda.set_device(int(config.ed_gpu))
    use_gpu = True ## todo

pre_embed = None
# if config.use_pre_embedding:
#     pre_embed = load_pre_embedding('../pre_embedding/sgns.baidubaike.bigram-char', word2id, config.embed_size)

model = Intent_Model(len(word2id), config.embed_size, config.hidden_size, 13, pre_embed, config.dropout, use_gpu)

if os.path.exists(config.model_path+'/ed_model/model.ckpt'):
    model.load_state_dict(torch.load(config.model_path+'/ed_model/model.ckpt'))
    print('load model state dict successful!')

if use_gpu:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)#, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
best_dev_acc = -1.0
earlystopping = EarlyStopping(config.delta, config.earlystop)
stop_flag = False
t_loss = 0.
b_loss = 0.
time1 = time.time()
for epoch in range(config.ed_epochs):
    np.random.shuffle(train_data)
    for i, data in enumerate(train_data):
        text_ids = data['text_ids']
        # mention_positions = data['mention_position']
        label = data['label']
        label = Variable(torch.LongTensor([label]))
        text_ids = Variable(torch.LongTensor(text_ids))


        if use_gpu:
            score = model(text_ids.cuda())
            label = label.cuda()
        else:
            score = model(text_ids)
        el_loss = criterion(score, label)
        b_loss += el_loss
        t_loss += el_loss.data

        if (i+1) % config.batch_size == 0:

            b_loss = b_loss / config.batch_size
            b_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()
            b_loss = 0.
            model.zero_grad()


        if (i + 1) % 100 == 0:
            print("Epoch: {}/{}, step: {}/{}, Loss: {}, Time:{}".format(epoch + 1, config.ed_epochs, i + 1, len(train_data), t_loss, time.time()-time1))
            time1 = time.time()
            stop_flag = earlystopping(t_loss)

            t_loss = 0.

        if (i + 1) % 1000 == 0:
            model.train(False)

            with torch.no_grad():
                dev_acc = evaluate(model, dev_data, use_gpu)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), config.model_path+'/model.ckpt')

            model.train(True)

            print("dev_acc:{:.4f}".format(dev_acc))

t_end = time.time()
print('best dev acc:{:.4f}'.format(best_dev_acc))
print('Train end! cost time:{}'.format(t_end - t_start))
