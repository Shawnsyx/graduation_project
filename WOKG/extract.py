#!/usr/bin/python
# -*- coding:utf8 -*-
## todo
# @Time    : 2020/3/27 17:10
# @Author  : cdtang

import sys
import csv
import json
from collections import defaultdict


def load_json_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))


def write_csv_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


def write_txt_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in data:
            f.write(i+"\n")

#value = "http://www.w3.org/2000/01/rdf-schema#value"
label = "http://www.w3.org/2000/01/rdf-schema#label"
alias = "http://www.w3.org/2000/01/rdf-schema#alias"
type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
nameIndividual = "http://www.w3.org/2002/07/owl#NamedIndividual"

label_result = []
alias_result = []
type_result = []
spo = []
entity = []
value_entity = []
relation = []

# test_file = open('./test.csv', 'w', encoding='utf-8', newline='') todo delete
# writer = csv.writer(test_file, delimiter=',')
# data = [[1,2,3], [4,5,6]]
# writer.writerows(data)
# test_file.close()

all_entity = []
with open('./all-spo.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for index, line in enumerate(reader):
        if index > 0:
            line = line[:-1]
            line[2] = ','.join(line[2:])
            line = line[:3]
            for i in range(len(line)):
                line[i] = line[i].strip().replace('"', '')

            # 分别将三元组加入 label, alias, spo
            if line[1] == label:
                label_result.append(line)
            elif line[1] == alias:
                alias_result.append(line)
            elif line[2] != nameIndividual:
                spo.append(line)
                if line[1] == type and line[2] not in type_result:
                    type_result.append(line[2])
            # 选出谓语
            if line[1] not in relation:
                relation.append(line[1])
            # 选出头实体
            if line[0] not in entity:
                entity.append(line[0])
            # 选出尾实体，判断是不是数字型实体
            if line[2] != nameIndividual and line[2].startswith("http"):
                if line[2] not in entity:
                    entity.append(line[2])
            elif line[1] != type and line[1] != label and line[1] != alias:
                if line[2] not in value_entity:
                    value_entity.append(line[2])
                # else:
                #     print("value_entity", line)
            if line[0] not in all_entity:
                all_entity.append(line[0])
            if line[2] not in all_entity:
                all_entity.append(line[2])
        # print(index, line)
        # writer.writerow(line)
        # if index > 100:
        #     sys.exit(0)

# test_file.close()
# print(len(all_entity))
# write_csv_file("./data/label.csv", label_result)
# write_csv_file("./data/alias.csv", alias_result)
# write_csv_file("./data/spo.csv", spo)
# write_txt_file("./data/entity.txt", entity)
# write_txt_file("./data/relation.txt", relation)
# write_txt_file("./data/value_entity.txt", value_entity)
# write_txt_file("./data/type.txt", type_result)


instance = "http://xlore.org/olympic/instance"
id2entity = defaultdict(list)
for i in label_result:
    id2entity[i[0]].append(i[2])
for i in alias_result:
    id2entity[i[0]].append(i[2])
print("find no label entity")
no_label_entity = []
label_entity = []
for i in entity:
    if i not in id2entity:
        no_label_entity.append(i)
    else:
        label_entity.append(i)
no_label_entity = [
'http://www.w3.org/2002/07/owl#Class',#大类
#'http://www.w3.org/2002/07/owl#topDataProperty',
'http://www.w3.org/2002/07/owl#DatatypeProperty', #数据属性
#'http://www.w3.org/2000/01/rdf-schema#alias',
#'http://www.w3.org/2002/07/owl#AnnotationProperty',
'http://www.w3.org/2002/07/owl#ObjectProperty',
'b0',
'http://www.w3.org/2002/07/owl#Ontology'
]
# print(len(label_entity))
# print(len(entity))
id2relation = {
    #'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': [[]],
    'http://xlore.org/olympic/property/50': '项目',
    #'http://www.w3.org/2000/01/rdf-schema#label': [[]],
    'http://xlore.org/olympic/property/5': '成绩', # '奖牌榜',
    'http://xlore.org/olympic/property/51': '吉祥物',
    #'http://www.w3.org/2000/01/rdf-schema#alias': [[]],
    'http://xlore.org/olympic/property/52': '比赛场馆',
    'http://xlore.org/olympic/property/1': '项目',
    'http://xlore.org/olympic/property/17': '参赛国家',
    'http://xlore.org/olympic/property/4': '第三名',
    'http://xlore.org/olympic/property/3': '第二名',
    'http://xlore.org/olympic/property/2': '第一名',
    #'http://www.w3.org/2000/01/rdf-schema#subClassOf': '包括', # 只有16中运动包含这个关系，将其转化为类型
    #'http://www.w3.org/2000/01/rdf-schema#subPropertyOf': [[]]
}

clean_id2entity = {}
for i, j in id2entity.items():
    if i not in id2relation:
        clean_id2entity[i] = j
    # else:
    #     print(i, j)
# print(len(clean_id2entity))
# for i in relation:
#     id2relation[i].append(id2entity[i])
# print(id2relation)


id2type = {
    'http://xlore.org/concept10': '场馆',
    'http://xlore.org/concept2': '参赛人员',
    'http://xlore.org/concept3': '冬季奥林匹克运动会', #'冬奥会',
    'http://xlore.org/sport15':  '速度滑冰', #'speed-skating',
    'http://xlore.org/sport3':  '雪车', #'bobsleigh',
    'http://xlore.org/sport1': '高山滑雪', #'alpine-skiing',
    'http://xlore.org/sport4': '越野滑雪', #'cross-country-skiing',
    'http://xlore.org/sport9': '雪橇', #'luge',
    'http://xlore.org/sport8': '冰球', #'ice-hockey',
    'http://xlore.org/sport10': '北欧两项', #'nordic-combined',
    #'http://www.w3.org/2002/07/owl#Class': [[]], # 作为实体， 其所有关系是类型
    'http://xlore.org/sport2': '冬季两项', #'biathlon',
    'http://xlore.org/sport6': '花样滑冰', #'figure-skating',
    'http://xlore.org/sport7':  '自由式滑雪', #'freestyle-skiing',
    #'http://www.w3.org/2002/07/owl#DatatypeProperty': [[]],# 作为实体， 其所有关系是类型
    'http://xlore.org/sport12': '钢架雪车', #'skeleton',
    'http://xlore.org/sport14': '单板滑雪', #'snowboard'，
    'http://xlore.org/sport13': '跳台滑雪', #'ski-jumping',
    'http://xlore.org/sport11': '短道速滑', #'short-track-speed-skating',
    'http://xlore.org/sport5': '冰壶', #'curling',
    #'http://www.w3.org/2002/07/owl#AnnotationProperty': [[]], # 作为实体， 其所有关系是类型
    #'http://www.w3.org/2002/07/owl#ObjectProperty': [[]], # 作为实体， 其所有关系是类型
    'http://xlore.org/sport16': '军事巡逻', #'military-patrol',
    #'http://www.w3.org/2002/07/owl#Ontology': [[]] # 作为实体， 其所有关系是类型
    "http://xlore.org/concept1": '比赛项目'
}

# 将id2type， id2relation， clean_id2entity写入txt文件
# write_json_file('./data/id2type.txt', id2type)
# write_json_file('./data/id2relation.txt', id2relation)
# write_json_file('./data/id2entity.txt', clean_id2entity)
# for i in type_result:
#     id2type[i].append(id2entity[i])
# print(id2type)



# build KG
type1 = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
id2name = load_json_file("./data/id2name.txt")
id2entity = load_json_file("./data/id2entity.txt")
id2relation = load_json_file("./data/id2relation.txt")
id2type = load_json_file("./data/id2type.txt")
print(len(id2name), len(id2type), len(id2entity), len(id2relation))

WOKG = {}
for i, j in id2name.items():
    my_type = ''
    object_data = []
    subject_data = []
    id = i
    name = j[0]
    alias = []
    for alia in id2entity[id]:
        if alia != name:
            alias.append(alia)
    for line in spo:
        if line[1] in id2relation:
            if line[0] == id:
                object_data.append(line)
            if line[2] == id:
                subject_data.append(line)
        else:
            if line[0] == id and (line[1] == type or line[1] == type1):
                if line[2] in id2type:
                    my_type = line[2]
    # my_type = my_type.strip().split(' ')
    # if len(my_type) != 1:
    #     print(id, my_type)
    WOKG[id] = {
        "name": name,
        "alias": alias,
        "type": my_type,
        "object_data": object_data,
        "subject_data": subject_data
    }

write_json_file("./data/WOKG.txt", WOKG)

