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

def produce_tag():
    list = ['START', 'B', 'I', 'O', 'STOP']
    tag2id = {t: i for i, t in enumerate(list)}
    id2tag = {i: t for i, t in enumerate(list)}
    return tag2id, id2tag


def read_data(data_path):

    data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        all_json_data = json.load(f)
        for k, v in all_json_data.items():

            text_id = k
            text = v['text']
            text = clear_entity(text)
            mention_data = v['mention_data']
            tags = get_bio_list(text, mention_data)
            data.append({
                'text_id': text_id,
                'text': text,
                'tags': tags,
                'mention_data': mention_data
            })

    return data

def get_bio_list(text, mention_data):
    tags = ['O']*len(text)
    for i in mention_data:
        mention = i[1]
        offset = int(i[0])
        start = True
        for j in range(offset, offset+len(mention)):
            if start:
                tags[j] = 'B'
                start = False
            else:
                tags[j] = 'I'
    return tags


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
    print("Found %i words, one pad, one unk" % len(word_dic))
    return word_dic, word2id, id2word


def prepare_ner_data(datas, word2id, tag2id):
    print(len(datas))
    for data in datas:
        text = data['text']
        tags = data['tags']

        text_ids = [word2id[i] if i in word2id else word2id['<UNK>'] for i in text]
        tag_ids = [tag2id[i] for i in tags]
        if len(text_ids) != len(tags) or len(tags) != len(tag_ids) or len(text_ids) != len(text):
            print(text, text_ids, tags, tag_ids)
        data['text_ids'] = text_ids
        data['tag_ids'] = tag_ids

def split_data(data_path):
    all_train = []
    with open(data_path, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for k, v in all.items():
            v["text_id"] = k
            all_train.append(v)

    print(len(all_train))
    np.random.shuffle(all_train)

    dev = all_train[:3000]
    test = all_train[3000:6000]
    train = all_train[6000:]
    print(len(train), len(dev), len(test))
    dis = [0]*20
    with open('./data/train.txt', 'w', encoding='utf-8') as f:
        for line in train:
            t = int(line["text_id"]) // 1000
            dis[t] += 1
            f.write(json.dumps(line, ensure_ascii=False)+"\n")
        print(dis)

    dis = [0] * 20
    with open('./data/dev.txt', 'w', encoding='utf-8') as f:
        for line in dev:
            t = int(line["text_id"]) // 1000
            dis[t] += 1
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(dis)

    dis = [0] * 20
    with open('./data/test.txt', 'w', encoding='utf-8') as f:
        for line in test:
            t = int(line["text_id"]) // 1000
            dis[t] += 1
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(dis)


if __name__ == '__main__':

    # split_data('./data/all_ner_data.txt')

    train_data_path = './data/all_ner_data.txt'

    tag2id, id2tag = produce_tag()

    train_data = read_data(train_data_path)
    with open('./data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    dev_data = train_data
    with open('./data/dev_data.pkl', 'wb') as f:
        pickle.dump(dev_data, f)

    word_dic, word2id, id2word = word_vocab(train_data_path)
    word_tag = {
        'tag2id': tag2id,
        'id2tag': id2tag,
        'word2id': word2id,
        'id2word': id2word
    }
    with open('./data/word_tag.pkl', 'wb') as f:
        pickle.dump(word_tag, f)
