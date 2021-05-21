  #!/usr/bin/python
# -*- coding:utf8 -*-

# @Time    : 2020/3/29 22:49
# @Author  : cdtang

import json
from collections import defaultdict


def load_json_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

WOKG = load_json_file("./data/WOKG.txt")
id2entity = load_json_file("./data/id2entity.txt")
id2name = load_json_file("./data/id2name.txt")
id2relation = {
    'http://xlore.org/olympic/property/50': '项目',
    'http://xlore.org/olympic/property/5': '成绩', # '奖牌榜',
    'http://xlore.org/olympic/property/51': '吉祥物',
    'http://xlore.org/olympic/property/52': '比赛场馆',
    'http://xlore.org/olympic/property/1': '项目',
    'http://xlore.org/olympic/property/17': '参赛国家',
    'http://xlore.org/olympic/property/4': '第三名',
    'http://xlore.org/olympic/property/3': '第二名',
    'http://xlore.org/olympic/property/2': '第一名',
    #'http://www.w3.org/2000/01/rdf-schema#subClassOf': '包括', # 只有16中运动包含这个关系，将其转化为类型
}

type2entity = {
    'http://xlore.org/concept10': [],
    'http://xlore.org/concept2': [],
    'http://xlore.org/concept3': [], #'冬奥会',
    'http://xlore.org/sport15':  [], #'speed-skating',
    'http://xlore.org/sport3':  [], #'bobsleigh',
    'http://xlore.org/sport1': [], #'alpine-skiing',
    'http://xlore.org/sport4': [], #'cross-country-skiing',
    'http://xlore.org/sport9': [], #'luge',
    'http://xlore.org/sport8': [], #'ice-hockey',
    'http://xlore.org/sport10': [], #'nordic-combined',
    'http://xlore.org/sport2': [], #'biathlon',
    'http://xlore.org/sport6': [], #'figure-skating',
    'http://xlore.org/sport7':  [], #'freestyle-skiing',
    'http://xlore.org/sport12': [], #'skeleton',
    'http://xlore.org/sport14': [], #'snowboard'，
    'http://xlore.org/sport13': [], #'ski-jumping',
    'http://xlore.org/sport11': [], #'short-track-speed-skating',
    'http://xlore.org/sport5': [], #'curling',
    'http://xlore.org/sport16': [], #'military-patrol',
    "http://xlore.org/concept1": [],
    "other": []
}

def build_type2entity():
    for k, v in WOKG.items():
        typeid = v["type"]
        if typeid in type2entity:
            type2entity[typeid].append(k)
        else:
            type2entity["other"].append(k)

    write_json_file("./data/type2entity.txt", type2entity)

type2entity = load_json_file("./data/type2entity.txt")

def build_187_178():
    WO_entity = type2entity["http://xlore.org/concept3"]
    data178 = {}
    data187 = {}
    textid = 1
    for id in WO_entity:
        my_spo = WOKG[id]["object_data"]
        my_alias = id2entity[id]
        my_mascot = "不知道"
        my_country = "不知道"
        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/51': # 吉祥物
                my_mascot = spo[2]
            if spo[1] == 'http://xlore.org/olympic/property/17': # 参赛国家
                my_country = spo[2]
        for alias in my_alias:
            one187 = [[alias+"的吉祥物是什么？", [[0, id, alias]], 0],
                      [my_mascot],
                      ["有哪些参赛国家？", [], 1],
                      [my_country]
            ]
            one178 = [[alias+"有哪些参赛国家？", [[0, id, alias]], 1],
                      [my_country],
                      ["吉祥物是什么？", [], 0],
                      [my_mascot]
            ]
            data178[textid] = one178
            data187[textid] = one187
            textid += 1
    write_json_file("./data/dialog/data178.txt", data178)
    write_json_file("./data/dialog/data187.txt", data187)
    print("build 187, 178 over")



def build_159():
    WO_entity = type2entity["http://xlore.org/concept3"]
    data159 = {}
    textid = 1
    for id in WO_entity:
        my_spo = WOKG[id]["object_data"]
        my_alias = id2entity[id]
        my_venue = []
        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/52':
                my_venue.append(spo[2])

        if len(my_venue) == 0:
            for alias in my_alias:
                one159 = [[alias + "有哪些比赛场馆？", [[0, id, alias]], 2],
                          ["不知道"]
                          ]
                data159[textid] = one159
                textid += 1
        else:
            venue_str = ','.join([id2name[i][0] for i in my_venue])
            for alias in my_alias:
                for venue_id in my_venue:
                    venue_name = id2name[venue_id][0]
                    my_game = "不知道"
                    venue_spo = WOKG[venue_id]["object_data"]
                    for spo in venue_spo:
                        if spo[1] == 'http://xlore.org/olympic/property/50':
                            my_game = spo[2]
                    one159 = [[alias + "有哪些比赛场馆？", [[0, id, alias]], 2],
                              [venue_str],
                              [venue_name+"举办过哪些比赛？", [[0,venue_id, venue_name]], 11],
                              [my_game]
                              ]
                    data159[textid] = one159
                    textid += 1
    write_json_file("./data/dialog/data159.txt", data159)
    print("build 159 over")

sport2game = {
    'http://xlore.org/sport15':  [],
    'http://xlore.org/sport3':  [],
    'http://xlore.org/sport1': [],
    'http://xlore.org/sport4': [],
    'http://xlore.org/sport9': [],
    'http://xlore.org/sport8': [],
    'http://xlore.org/sport10': [],
    'http://xlore.org/sport2': [],
    'http://xlore.org/sport6': [],
    'http://xlore.org/sport7':  [],
    'http://xlore.org/sport12': [],
    'http://xlore.org/sport14': [],
    'http://xlore.org/sport13': [],
    'http://xlore.org/sport11': [],
    'http://xlore.org/sport5': [],
    'http://xlore.org/sport16': []
}
def build_43():
    data43 = {}
    textid = 1
    for sport in sport2game.keys():
        sport_name = id2name[sport][0]
        games = type2entity[sport]
        game_name = []
        for gameid in games:
            name = id2name[gameid][0]
            if name not in game_name:
                game_name.append(name)

        one43 = [
            [sport_name + "有哪些子项目？", [[0, sport, sport_name]], 4],
            [','.join(game_name)]
        ]
        data43[textid] = one43
        textid += 1
        sport2game[sport] = game_name
    write_json_file("./data/dialog/data43.txt", data43)

    write_json_file("./data/dialog/sport2game.txt", sport2game)
    print("build 43 over")

sport2game = load_json_file("./data/dialog/sport2game.txt")

def build_14326():
    WO_entity = type2entity["http://xlore.org/concept3"]
    data14326 = {}
    textid = 1
    for id in WO_entity:
        my_spo = WOKG[id]["object_data"]
        my_alias = id2entity[id]
        my_sports = []
        my_games = []
        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/1':
                my_games.append(spo[2])
                s = WOKG[spo[2]]["type"]
                if s not in my_sports:
                    my_sports.append(s)
        my_sport_str = ','.join([id2name[s][0] for s in my_sports])
        if len(my_sports) == 0:
            one14326 = [
                [alias + "有哪些比赛项目？", [[0, id, alias]], 3],
                ["不知道"]
            ]
            data14326[textid] = one14326
            textid += 1
        for sportid in my_sports:
            for gameid in my_games:
                game_type = WOKG[gameid]["type"]
                if game_type == sportid:
                    game_spo = WOKG[gameid]["object_data"]
                    if len(game_spo) == 0:
                        one14326 = [
                            [alias + "有哪些比赛项目？", [[0, id, alias]], 3],
                            [my_sport_str],
                            [id2name[sportid][0] + "有哪些子项目？", [[0, sportid, id2name[sportid][0]]], 4],
                            [','.join(sport2game[sportid])],
                            [id2name[gameid][0] + "的第一名是谁？", [[0, gameid, id2name[gameid][0]]], 5],
                            ["不知道"]
                        ]
                        data14326[textid] = one14326
                        textid += 1
                        one14326 = [
                            [alias + "有哪些比赛项目？", [[0, id, alias]], 3],
                            [my_sport_str],
                            [id2name[sportid][0] + "有哪些子项目？", [[0, sportid, id2name[sportid][0]]], 4],
                            [','.join(sport2game[sportid])],
                            [id2name[gameid][0] + "的第二名是谁？", [[0, gameid, id2name[gameid][0]]], 6],
                            ["不知道"]
                        ]
                        data14326[textid] = one14326
                        textid += 1
                        one14326 = [
                            [alias + "有哪些比赛项目？", [[0, id, alias]], 3],
                            [my_sport_str],
                            [id2name[sportid][0] + "有哪些子项目？", [[0, sportid, id2name[sportid][0]]], 4],
                            [','.join(sport2game[sportid])],
                            [id2name[gameid][0] + "的第三名是谁？", [[0, gameid, id2name[gameid][0]]], 7],
                            ["不知道"]
                        ]
                        data14326[textid] = one14326
                        textid += 1
                    for g_spo in game_spo:
                        if g_spo[1] == 'http://xlore.org/olympic/property/2' or g_spo[1] == 'http://xlore.org/olympic/property/3' or g_spo[1] == 'http://xlore.org/olympic/property/4':
                            personid = g_spo[2]
                            per_spo = WOKG[personid]["object_data"]
                            per_records = []
                            for p_spo in per_spo:
                                if p_spo[1] == 'http://xlore.org/olympic/property/5':
                                    per_records.append(p_spo[2])
                            # if len(per_records) == 0:
                            #     print(personid)
                            for alias in my_alias:
                                ranking = "第一名"
                                intent = 5
                                if g_spo[1] == 'http://xlore.org/olympic/property/3':
                                    ranking = "第二名"
                                    intent = 6
                                elif g_spo[1] == 'http://xlore.org/olympic/property/4':
                                    ranking = "第三名"
                                    intent = 7
                                one14326 = [
                                    [alias+"有哪些比赛项目？", [[0, id, alias]], 3],
                                    [my_sport_str],
                                    [id2name[sportid][0]+"有哪些子项目？",[[0, sportid, id2name[sportid][0]]], 4],
                                    [','.join(sport2game[sportid])],
                                    [id2name[gameid][0]+"的"+ranking+"是谁？", [[0, gameid, id2name[gameid][0]]], intent],
                                    [id2name[personid][0]],
                                    ["他还有哪些成绩？", [], 8],
                                    [",".join(per_records)]
                                ]
                                data14326[textid] = one14326
                                textid += 1
    write_json_file("./data/dialog/data14326.txt", data14326)
    print("build 14326 over")

def build_1326():
    WO_entity = type2entity["http://xlore.org/concept3"]
    data1326 = {}
    textid = 1
    for id in WO_entity:
        my_spo = WOKG[id]["object_data"]
        my_alias = id2entity[id]
        my_sports = []
        my_games = []
        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/1':
                my_games.append(spo[2])
                s = WOKG[spo[2]]["type"]
                if s not in my_sports:
                    my_sports.append(s)
        my_sport_str = ','.join([id2name[s][0] for s in my_sports])
        if len(my_sports) > 0:
            for spo in my_spo:
                if spo[1] == 'http://xlore.org/olympic/property/1':
                    gameid = spo[2]
                    game_spo = WOKG[gameid]["object_data"]
                    sportid = WOKG[gameid]["type"]
                    for g_spo in game_spo:
                        if g_spo[1] == 'http://xlore.org/olympic/property/2' or g_spo[1] == 'http://xlore.org/olympic/property/3' or g_spo[1] == 'http://xlore.org/olympic/property/4':
                            personid = g_spo[2]
                            per_spo = WOKG[personid]["object_data"]
                            per_records = []
                            for p_spo in per_spo:
                                if p_spo[1] == 'http://xlore.org/olympic/property/5':
                                    per_records.append(p_spo[2])
                            for alias in my_alias:
                                ranking = "第一名"
                                intent = 5
                                if g_spo[1] == 'http://xlore.org/olympic/property/3':
                                    ranking = "第二名"
                                    intent = 6
                                elif g_spo[1] == 'http://xlore.org/olympic/property/4':
                                    ranking = "第三名"
                                    intent = 7
                                one1326 = [
                                    [alias+"有哪些比赛项目？", [[0, id, alias]], 3],
                                    [my_sport_str],
                                    [id2name[sportid][0]+id2name[gameid][0]+"的"+ranking+"是谁？",[[0, sportid, id2name[sportid][0]],[len(id2name[sportid][0]), gameid, id2name[gameid][0]]], intent],
                                    [id2name[personid][0]],
                                    ["他还有哪些成绩？", [], 8],
                                    [",".join(per_records)]
                                ]
                                data1326[textid] = one1326
                                textid += 1
    write_json_file("./data/dialog/data1326.txt", data1326)
    print("build 1326 over")

def build_519_591():
    data519 = {}
    data591 = {}
    textid = 1
    WO_entity = type2entity["http://xlore.org/concept10"]
    for id in WO_entity:
        ob_spo = WOKG[id]["object_data"]
        su_spo = WOKG[id]["subject_data"]
        my_games = "不知道"
        my_wo = "不知道"
        for spo in ob_spo:
            if spo[1] == "http://xlore.org/olympic/property/50":
                my_games = spo[2]
        for spo in su_spo:
            if spo[1] == "http://xlore.org/olympic/property/52":
                my_wo = spo[0]
        one591 = [
            [id2name[id][0]+"举办过哪些比赛？", [[0,id, id2name[id][0]]], 11],
            [my_games],
            ["举办过第几届冬奥会？",[], 12],
            [id2name[my_wo][0]]
        ]
        one519 = [
            [id2name[id][0] + "举办过第几届冬奥会？", [[0, id, id2name[id][0]]], 12],
            [id2name[my_wo][0]],
            ["举办过哪些比赛？", [], 11],
            [my_games]
        ]
        data591[textid] = one591
        data519[textid] = one519
        textid += 1
    write_json_file("./data/dialog/data591.txt", data591)
    write_json_file("./data/dialog/data519.txt", data519)
    print("build 519, 591 over")

def build_5178_5187():
    data5178 = {}
    data5187 = {}
    textid = 1
    WO_entity = type2entity["http://xlore.org/concept10"]
    for id in WO_entity:
        su_spo = WOKG[id]["subject_data"]
        woid = ""
        for spo in su_spo:
            if spo[1] == "http://xlore.org/olympic/property/52":
                woid = spo[0]
        my_spo = WOKG[woid]["object_data"]
        my_mascot = "不知道"
        my_country = "不知道"
        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/51':
                my_mascot = spo[2]
            if spo[1] == 'http://xlore.org/olympic/property/17':
                my_country = spo[2]

        one5187 = [[id2name[id][0] + "举办过第几届冬奥会？", [[0, id, id2name[id][0]]], 12],
                      [id2name[woid][0]],
                      ["这届冬奥会的吉祥物是什么？", [[2, woid, "冬奥会"]], 0],
                      [my_mascot],
                      ["有哪些参赛国家？", [], 1],
                      [my_country]
                      ]
        one5178 = [[id2name[id][0] + "举办过第几届冬奥会？", [[0, id, id2name[id][0]]], 12],
                      [id2name[woid][0]],
                      ["这届冬奥会有哪些参赛国家？", [[2, woid, "冬奥会"]], 1],
                      [my_country],
                      ["吉祥物是什么？", [], 0],
                      [my_mascot]
                      ]
        data5178[textid] = one5178
        data5187[textid] = one5187
        textid += 1
    write_json_file("./data/dialog/data5178.txt", data5178)
    write_json_file("./data/dialog/data5187.txt", data5187)
    print("build 5178, 5187 over")

def build_236_263():
    data236 = {}
    data263 = {}
    textid = 1
    WO_entity = type2entity["http://xlore.org/concept2"]
    for id in WO_entity:
        su_spo = WOKG[id]["subject_data"]
        ob_spo = WOKG[id]["object_data"]
        per_records = []
        per_sports = []
        for spo in ob_spo:
            if spo[1] == "http://xlore.org/olympic/property/5":
                per_records.append(spo[2])
        for spo in su_spo:
            if spo[1] == "http://xlore.org/olympic/property/4" or spo[1] == "http://xlore.org/olympic/property/3" or spo[1] == "http://xlore.org/olympic/property/2":
                per_game = spo[0]
                per_sport = id2name[WOKG[per_game]["type"]][0]
                if per_sport not in per_sports:
                    per_sports.append(per_sport)
        one236 = [
            [id2name[id][0] +"取得了哪些成绩？", [[0, id, id2name[id][0]]], 8],
            [','.join(per_records)],
            ["参加了哪些比赛？", [], 9],
            [','.join(per_sports)]
        ]
        one263 = [
            [id2name[id][0] + "参加了哪些比赛？", [[0, id, id2name[id][0]]], 9],
            [','.join(per_sports)],
            ["取得了哪些成绩？", [], 8],
            [','.join(per_records)]
        ]
        data236[textid] = one236
        data263[textid] = one263
        textid += 1
    write_json_file("./data/dialog/data236.txt", data236)
    write_json_file("./data/dialog/data263.txt", data263)
    print("build 236, 263 over")

def build_21():
    data21 = {}
    textid = 1
    WO_entity = type2entity["http://xlore.org/concept2"]
    for id in WO_entity:
        su_spo = WOKG[id]["subject_data"]
        per_wo = []
        for spo in su_spo:
            if spo[1] == "http://xlore.org/olympic/property/4" or spo[1] == "http://xlore.org/olympic/property/3" or \
                    spo[1] == "http://xlore.org/olympic/property/2":
                per_game = spo[0]
                per_game_spo = WOKG[per_game]["subject_data"]
                for game_spo in per_game_spo:
                    if game_spo[1] == "http://xlore.org/olympic/property/1":
                        wo = id2name[game_spo[0]][0]
                        if wo not in per_wo:
                            per_wo.append(wo)
        one21 = [
            [id2name[id][0]+"参加了哪几届冬奥会？", [[0, id, id2name[id][0]]], 10],
            [','.join(per_wo)]
        ]
        data21[textid] = one21
        textid += 1
    write_json_file("./data/dialog/data21.txt", data21)
    print("build 21 over")

def build_132178_132187():
    WO_entity = type2entity["http://xlore.org/concept3"]
    data132178 = {}
    data132187 = {}
    textid = 1
    for id in WO_entity:
        my_spo = WOKG[id]["object_data"]
        my_alias = id2entity[id]

        my_mascot = "不知道"
        my_country = "不知道"
        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/51':
                my_mascot = spo[2]
            elif spo[1] == 'http://xlore.org/olympic/property/17':
                my_country = spo[2]
            elif spo[1] == 'http://xlore.org/olympic/property/1':
                gameid = spo[2]
                game_name = id2name[gameid][0]
                game_type = WOKG[gameid]["type"]
                game_type_name = id2name[game_type][0]
                game_spo = WOKG[gameid]["object_data"]
                for g_spo in game_spo:
                    if g_spo[1] == 'http://xlore.org/olympic/property/2' or g_spo[1] == 'http://xlore.org/olympic/property/3' or g_spo[1] == 'http://xlore.org/olympic/property/4':
                        personid = g_spo[2]
                        for alias in my_alias:
                            ranking = "第一名"
                            intent = 5
                            if g_spo[1] == 'http://xlore.org/olympic/property/3':
                                ranking = "第二名"
                                intent = 6
                            elif g_spo[1] == 'http://xlore.org/olympic/property/4':
                                ranking = "第三名"
                                intent = 7
                            one132187 = [
                                [alias+"中"+game_type_name+game_name+"的"+ranking+"是谁？", [[0, id, alias], [len(alias+"中"), game_type, game_type_name], [len(alias+"中"+game_type_name), gameid, game_name]], intent],
                                [id2name[personid][0]],
                                ["这届冬奥会的吉祥物是什么",[[2, id , "冬奥会"]], 0],
                                [my_mascot],
                                ["有哪些参赛国家？", [], 1],
                                [my_country]
                            ]
                            one132178 = [
                                [alias + "中" + game_type_name + game_name + "的" + ranking + "是谁？",[[0, id, alias], [len(alias + "中"), game_type, game_type_name], [len(alias + "中" + game_type_name), gameid, game_name]], intent],
                                [id2name[personid][0]],
                                ["这届冬奥会的有哪些参赛国家？", [[2, id, "冬奥会"]], 1],
                                [my_country],
                                ["吉祥物是什么？", [], 0],
                                [my_mascot]
                            ]
                            data132178[textid] = one132178
                            data132187[textid] = one132187
                            textid += 1
    write_json_file("./data/dialog/data132178.txt", data132178)
    write_json_file("./data/dialog/data132187.txt", data132187)
    print("build 132178, 132187 over")

def build_1326_13222():
    WO_entity = type2entity["http://xlore.org/concept3"]
    data1326 = {}
    data13222 = {}
    textid1 = 1
    textid2 = 1
    for id in WO_entity:
        my_spo = WOKG[id]["object_data"]
        my_alias = id2entity[id]

        for spo in my_spo:
            if spo[1] == 'http://xlore.org/olympic/property/1':
                gameid = spo[2]
                game_name = id2name[gameid][0]
                game_type = WOKG[gameid]["type"]
                game_type_name = id2name[game_type][0]
                game_spo = WOKG[gameid]["object_data"]
                per_1 = ""
                per_2 = ""
                per_3 = ""
                for g_spo in game_spo:
                    if g_spo[1] == 'http://xlore.org/olympic/property/2':
                        per_1 = g_spo[2]
                    elif g_spo[1] == 'http://xlore.org/olympic/property/3':
                        per_2 = g_spo[2]
                    elif g_spo[1] == 'http://xlore.org/olympic/property/4':
                        per_3 = g_spo[2]
                per1_name = id2name[per_1][0] if per_1 != "" else "不知道"
                per2_name = id2name[per_2][0] if per_2 != "" else "不知道"
                per3_name = id2name[per_3][0] if per_3 != "" else "不知道"
                for alias in my_alias:
                    if per_1 != "":
                        per1_spo = WOKG[per_1]["object_data"]
                        per_records = []
                        for per_spo in per1_spo:
                            if per_spo[1] == "http://xlore.org/olympic/property/5":
                                per_records.append(per_spo[2])
                        one1326 = [
                            [alias + "中" + game_type_name + game_name + "的第一名是谁？",[[0, id, alias], [len(alias + "中"), game_type, game_type_name],[len(alias + "中" + game_type_name), gameid, game_name]], 5],
                            [per1_name],
                            ["他还有哪些成绩？", [], 8],
                            [','.join(per_records)]
                        ]
                        data1326[textid1] = one1326
                        textid1 += 1
                    else:
                        one1326= [
                            [alias + "中" + game_type_name + game_name + "的第一名是谁？",
                             [[0, id, alias], [len(alias + "中"), game_type, game_type_name],
                              [len(alias + "中" + game_type_name), gameid, game_name]], 5],
                            [per1_name]
                        ]
                        data1326[textid1] = one1326
                        textid1 += 1

                    if per_2 != "":
                        per2_spo = WOKG[per_2]["object_data"]
                        per_records = []
                        for per_spo in per2_spo:
                            if per_spo[1] == "http://xlore.org/olympic/property/5":
                                per_records.append(per_spo[2])
                        one1326 = [
                                [alias + "中" + game_type_name + game_name + "的第二名是谁？",
                                 [[0, id, alias], [len(alias + "中"), game_type, game_type_name],
                                  [len(alias + "中" + game_type_name), gameid, game_name]], 6],
                                [per2_name],
                                ["他还有哪些成绩？", [], 8],
                                [','.join(per_records)]
                            ]
                        data1326[textid1] = one1326
                        textid1 += 1
                    else:
                        one1326 = [
                                [alias + "中" + game_type_name + game_name + "的第二名是谁？",
                                 [[0, id, alias], [len(alias + "中"), game_type, game_type_name],
                                  [len(alias + "中" + game_type_name), gameid, game_name]], 6],
                                [per2_name]
                            ]
                        data1326[textid1] = one1326
                        textid1 += 1
                    if per_3 != "":
                        per3_spo = WOKG[per_3]["object_data"]
                        per_records = []
                        for per_spo in per3_spo:
                            if per_spo[1] == "http://xlore.org/olympic/property/5":
                                per_records.append(per_spo[2])
                        one1326 = [
                                [alias + "中" + game_type_name + game_name + "的第三名是谁？",
                                 [[0, id, alias], [len(alias + "中"), game_type, game_type_name],
                                  [len(alias + "中" + game_type_name), gameid, game_name]], 7],
                                [per3_name],
                                ["他还有哪些成绩？", [], 8],
                                [','.join(per_records)]
                            ]
                        data1326[textid1] = one1326
                        textid1 += 1
                    else:
                        one1326 = [
                                [alias + "中" + game_type_name + game_name + "的第三名是谁？",
                                 [[0, id, alias], [len(alias + "中"), game_type, game_type_name],
                                  [len(alias + "中" + game_type_name), gameid, game_name]], 7],
                                [per3_name]
                            ]
                        data1326[textid1] = one1326
                        textid1 += 1

                    one13222 = [
                                [alias + "中" + game_type_name + game_name + "的第一名是谁？",[[0, id, alias], [len(alias + "中"), game_type, game_type_name],
                                  [len(alias + "中" + game_type_name), gameid, game_name]], 5],
                                [per1_name],
                                ["第二名是谁？", [], 6],
                                [per2_name],
                                ["第三名是谁？", [], 7],
                                [per3_name]
                            ]
                    data13222[textid2] = one13222
                    textid2 += 1
    write_json_file("./data/dialog/data13222.txt", data13222)
    write_json_file("./data/dialog/data1326.txt", data1326)
    print("build 1326, 13222 over")

build_187_178()
build_159()
build_43()
build_14326()
build_1326()
build_519_591()
build_5178_5187()
build_236_263()
build_21()
build_1326_13222()
build_132178_132187()