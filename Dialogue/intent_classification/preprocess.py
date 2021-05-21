# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: preprocess.py
@time: 19-11-11 下午10:22

"""
import numpy as np
import json
from collections import defaultdict
import pickle
from config import make_config
from random import choice
import time

config = make_config()



def clear_entity(text):
    # ban = {
    #     '，': ',',
    #     '·': '•',
    #     '：': ':',
    #     '！': '!',
    # }
    # for i in ban:
    #     if i in text:
    #         text = text.replace(i, ban[i])
    return text.lower()



def read_data(data_path):

    data = []
    dis = [0]*13
    with open(data_path, 'r', encoding='utf-8') as f:
        all_json_data = json.load(f)
        for k, v in all_json_data.items():
            text_id = k
            text = v['text']
            text = clear_entity(text)
            intent = v['intent']
            data.append({
                'text_id': text_id,
                'text': text,
                'intent': intent
            })

            dis[int(intent)] += 1
    print(dis) #[96, 96, 93, 89, 32, 4022, 4023, 4023, 1481, 1480, 1479, 48, 48]

    return data



def split_data(data_path):
    all_train = []
    with open(data_path, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for k, v in all.items():
            v["text_id"] = k
            all_train.append(v)
    print(len(all_train))
    np.random.shuffle(all_train)

    dev = all_train[:12000]
    test = all_train[12000:24000]
    train = all_train[24000:]
    print(len(train), len(dev), len(test))

    with open('./data/train.txt', 'w', encoding='utf-8') as f:
        for line in train:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")

    with open('./data/dev.txt', 'w', encoding='utf-8') as f:
        for line in dev:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")

    with open('./data/test.txt', 'w', encoding='utf-8') as f:
        for line in test:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")
    print("split over!")

def build_dictionary(word_list):
    dic = {}
    for line in word_list:
        for i in line:
            dic[i] = dic.get(i, 0) + 1
    return dic

def word_vocab(train_data_path):
    word_list = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k, v in data.items():
            text = v['text']
            text = clear_entity(text)
            word_list.append(text)
    word_dic = build_dictionary(word_list)
    word_dic['<PAD>'] = 1000000001
    word_dic['<UNK>'] = 1000000002
    # print(len(word_dic))
    sorted_word_dic = sorted(word_dic.items(), key=lambda x: (-x[1], x[0]))
    # sorted_word_dic = [(k,v) for k,v in sorted_word_dic if v >= 3]
    id2word = {k: v[0] for k, v in enumerate(sorted_word_dic)}
    word2id = {v: k for k, v in id2word.items()}
    print("Found %i words, one pad, one unk" % len(word_dic)) #Found 2775 words, one pad, one unk
    return word_dic, word2id, id2word



def prepare_intent_data(datas, word2id):
    print('prepare intent data: ', len(datas))
    res = []
    for data in datas:

        text = data['text']
        text_ids = [word2id[i] if i in word2id else word2id['<UNK>'] for i in text]
        label = data['intent']

        temp_data = {}
        temp_data['text'] = text
        temp_data['text_ids'] = text_ids
        temp_data['label'] = label
        res.append(temp_data)

    return res



if __name__ == '__main__':

    # split_data('./data/all_intent_data.txt')

    train_data_path = './data/all_intent_data.txt'

    word_dic, word2id, id2word = word_vocab(train_data_path)
    word = {
        'word2id': word2id,
        'id2word': id2word
    }
    with open('./data/word_dic.pkl', 'wb') as f:
        pickle.dump(word, f)

    train_data = read_data(train_data_path)
    t = time.time()
    train_data = prepare_intent_data(train_data, word2id)
    t1 = time.time()
    print("train prepare end, cost time", t1-t)
    with open('./data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    dev_data = train_data
    t2 = time.time()
    print("dev prepare end, cost time", t2 - t1)
    with open('./data/dev_data.pkl', 'wb') as f:
        pickle.dump(dev_data, f)



