#!/usr/bin/python
# -*- coding:utf8 -*-


import json

def read_json_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def normalize_text(text):
    return text.lower()

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