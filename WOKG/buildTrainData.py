#!/usr/bin/python
# -*- coding:utf8 -*-

# @Time    : 2020/4/3 5:06
# @Author  : cdtang

import json
import numpy as np
all_data = {}

def read_json_file(file_name, index):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k, v in data.items():
            index += 1
            all_data[index] = v
    return index

def write_json_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

def clear_all_data():

    for k, v in all_data.items():
        for i, value in enumerate(v):
            if i % 2 == 0:
                if len(value) == 3:
                    text = value[0]
                    if len(value[1]) > 0:
                        if type(value[1][0]).__name__ != "list":
                            print(k, v)
                    for l in value[1]:
                        start = l[0]
                        name = l[2]
                        if name != text[start:start+len(name)]:
                            print(k, v)
                else:
                    print(k, v)

def produce_ner_data(): #实体命名
    ner_data = {}
    ner_set_data = []
    ner_index = 0
    all = 0
    for k, v in all_data.items():
        all += len(v) / 2
        for i, value in enumerate(v):
            if i % 2 == 0:
                mention = []
                for m in value[1]:
                    mention.append([m[0], m[2]])
                one_data = [value[0], mention]
                if one_data not in ner_set_data:
                    ner_set_data.append(one_data)

    print("all ner data: ", all) #160594
    print(len(ner_set_data)) #17010
    for i, v in enumerate(ner_set_data):
        text = v[0]
        mention_data = []
        if len(v) == 2:
            mention_data = v[1]
        else:
            print(v)
        ner_data[ner_index] = {"text": text, "mention_data": mention_data}
        ner_index += 1
    write_json_data("./data/dialog/all_ner_data.txt", ner_data)
    print("ner data over")


def produce_intent_data(): # 实体链接
    intent_data = {}
    intent_set_data = []
    index = 0
    all = 0
    for k, v in all_data.items():
        all += len(v) / 2
        for i, value in enumerate(v):
            if i % 2 == 0:
                one_data = [value[0], value[2]]
                if one_data not in intent_set_data:
                    intent_set_data.append(one_data)
    print(all) #160594
    print(len(intent_set_data)) #17010
    for i, v in enumerate(intent_set_data):
        text = v[0]
        intent = v[1]
        intent_data[index] = {"text": text, "intent": intent}
        index += 1
    print("all intent data:", index) #17010
    write_json_data("./data/dialog/all_intent_data.txt", intent_data)
    print("el intent over")



def split_data(dialog_data):
    np.random.shuffle(dialog_data)
    dev = dialog_data[:30000]
    train = dialog_data[30000:]
    return train,dev


index = 0
# index = read_json_file("./data/dialog/data21.txt", index)
# index = read_json_file("./data/dialog/data43.txt", index)
# index = read_json_file("./data/dialog/data159.txt", index)
# index = read_json_file("./data/dialog/data178.txt", index)
# index = read_json_file("./data/dialog/data187.txt", index)
# index = read_json_file("./data/dialog/data236.txt", index)
# index = read_json_file("./data/dialog/data263.txt", index)
# index = read_json_file("./data/dialog/data519.txt", index)
# index = read_json_file("./data/dialog/data591.txt", index)
# index = read_json_file("./data/dialog/data1326.txt", index)
# index = read_json_file("./data/dialog/data5178.txt", index)
# index = read_json_file("./data/dialog/data5187.txt", index)
# index = read_json_file("./data/dialog/data13222.txt", index)
# index = read_json_file("./data/dialog/data14326.txt", index)
# index = read_json_file("./data/dialog/data132178.txt", index)
# index = read_json_file("./data/dialog/data132187.txt", index)
# print(index) # 55918
# write_json_data("./data/dialog/all_train_data.txt", all_data)

# read_json_file("./data/dialog/all_train_data.txt", index)
# clear_all_data()
# produce_ner_data()
# produce_intent_data()