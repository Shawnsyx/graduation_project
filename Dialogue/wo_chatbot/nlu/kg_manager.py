#!/usr/bin/python
# -*- coding:utf8 -*-


from collections import defaultdict
from utils import read_json_file

class KG_Manager(object):

    def __init__(self, config):
        self.id2name = read_json_file(config.id2name)
        for k in self.id2name:
            v = self.id2name[k][0]
            self.id2name[k] = v
        self.id2entity = read_json_file(config.id2entity)
        self.entity2id = defaultdict(list)
        for k, v in self.id2entity.items():
            for i in v:
                if k not in self.entity2id[i]:
                    self.entity2id[i].append(k)
        self.id2relation = read_json_file(config.id2relation)
        self.id2type = read_json_file(config.id2type)
        self.WOKG = read_json_file(config.WOKG)
        self.sport2game = read_json_file(config.sport2game)

        self.id2func = {
            0: self.ask_mascot,
            1: self.ask_country,
            2: self.ask_venue,
            3: self.ask_game,
            4: self.ask_sport,
            5: self.ask_first,
            6: self.ask_second,
            7: self.ask_third,
            8: self.ask_achievement,
            9: self.ask_athlete_sport,
            10: self.ask_athlete_wo,
            11: self.ask_venue_sport,
            12: self.ask_venue_wo
        }

    def clear_entity2id(self):
        for k in self.entity2id:
            if len(self.entity2id[k]) > 1:
                for i in self.entity2id[k]:
                    i_type = self.WOKG[i]["type"]
                    if i_type == "http://xlore.org/concept10" or i_type == "http://xlore.org/concept2" or \
                            i_type == "http://xlore.org/concept3" or i_type == "http://xlore.org/concept1" :
                        print("ids:", k, self.entity2id[k])


    def entity2type(self, entity):
        if entity not in self.entity2id:
            print("not exist ", entity)
            return None, entity
        else:
            ids = self.entity2id[entity]
            if len(ids) > 1:
                return "game", entity
            else:
                id = ids[0]
                type_id = self.WOKG[id]["type"]
                if type_id == "http://xlore.org/concept10":
                    return "venue", self.id2name[id]
                elif type_id == "http://xlore.org/concept2":
                    return "athlete", self.id2name[id]
                elif type_id == "http://xlore.org/concept3":
                    return "olympic", self.id2name[id]
                elif type_id == "http://xlore.org/concept1":
                    return "sport", self.id2name[id]
                else:
                    print("unknow type", type_id, entity)
                    return None, entity

    def ask_mascot(self, history):
        olympic = history["olympic"]
        if olympic == '':
            return "请问是哪一届冬奥会？"

        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/51":
                return object_data[2]
        return "不知道"


    def ask_country(self, history):
        olympic = history["olympic"]
        if olympic == '':
            return "请问是哪一届冬奥会？"

        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/17":
                country = object_data[2]
                c = country.split(',')
                c_length = str(len(c))
                return "一共有"+c_length+"个国家，分别为："+country
        return "不知道"

    def ask_venue(self, history):
        olympic = history["olympic"]
        if olympic == '':
            return "请问是哪一届冬奥会？"

        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        answer = []
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/52":
                answer.append(self.id2name[object_data[2]])
        if len(answer) == 0:
            return "不知道"
        return "一共有"+str(len(answer))+"个场馆，分别为："+','.join(answer)

    def ask_game(self, history):
        olympic = history["olympic"]
        if olympic == '':
            return "请问是哪一届冬奥会？"

        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        answer = []
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/1":
                game_id = object_data[2]
                sport_id = self.WOKG[game_id]["type"]
                sport_name = self.id2name[sport_id]
                if sport_name not in answer:
                    answer.append(sport_name)
        if len(answer) == 0:
            return "不知道"
        return "一共有" + str(len(answer)) + "比赛大类，分别为：" + ','.join(answer)

    def ask_sport(self, history):
        sport = history["sport"]
        if sport == "钢架雪车":
            sport = "雪车"
        sport_id = self.entity2id[sport][0]
        answer = self.sport2game[sport_id]
        if len(answer) == 0:
            return "不知道"
        return "一共有" + str(len(answer)) + "个子项目，分别为：" + ','.join(answer)

    def ask_first(self, history):
        olympic = history["olympic"]
        sport = history["sport"]
        game = history["game"]
        if olympic == '':
            return "请问是哪一届冬奥会的"+sport+game+"？"
        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        answer = ""
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/1":
                game_id = object_data[2]
                game_name = self.id2name[game_id]
                sport_id = self.WOKG[game_id]["type"]
                sport_name = self.id2name[sport_id]
                if sport_name == sport and game_name == game:
                    game_object_data = self.WOKG[game_id]["object_data"]
                    for o in game_object_data:
                        if o[1] == "http://xlore.org/olympic/property/2":
                            athlete_id = o[2]
                            answer = self.id2name[athlete_id]
        if answer == "":
            return "不知道"
        return answer

    def ask_second(self, history):
        olympic = history["olympic"]
        sport = history["sport"]
        game = history["game"]
        if olympic == '':
            return "请问是哪一届冬奥会的" + sport + game + "？"
        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        answer = ""
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/1":
                game_id = object_data[2]
                game_name = self.id2name[game_id]
                sport_id = self.WOKG[game_id]["type"]
                sport_name = self.id2name[sport_id]
                if sport_name == sport and game_name == game:
                    game_object_data = self.WOKG[game_id]["object_data"]
                    for o in game_object_data:
                        if o[1] == "http://xlore.org/olympic/property/3":
                            athlete_id = o[2]
                            answer = self.id2name[athlete_id]
        if answer == "":
            return "不知道"
        return answer

    def ask_third(self, history):
        olympic = history["olympic"]
        sport = history["sport"]
        game = history["game"]
        if olympic == '':
            return "请问是哪一届冬奥会的" + sport + game + "？"
        olympic_id = self.entity2id[olympic][0]
        olympic_object_data = self.WOKG[olympic_id]["object_data"]
        answer = ""
        for object_data in olympic_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/1":
                game_id = object_data[2]
                game_name = self.id2name[game_id]
                sport_id = self.WOKG[game_id]["type"]
                sport_name = self.id2name[sport_id]
                if sport_name == sport and game_name == game:
                    game_object_data = self.WOKG[game_id]["object_data"]
                    for o in game_object_data:
                        if o[1] == "http://xlore.org/olympic/property/4":
                            athlete_id = o[2]
                            answer = self.id2name[athlete_id]
        if answer == "":
            return "不知道"
        return answer

    def ask_achievement(self, history):
        athlete = history["athlete"]
        if athlete == '':
            return "请问是哪个运动员？"
        answer = []
        athlete_id = self.entity2id[athlete][0]
        athlete_subject_data = self.WOKG[athlete_id]["subject_data"]
        for subject_data in athlete_subject_data:
            game_id = subject_data[0]
            game_subject_data = self.WOKG[game_id]["subject_data"]
            for game_data in game_subject_data:
                o = game_data[0]
                if self.WOKG[o]["type"] == "http://xlore.org/concept3":
                    olympic_name = self.id2name[o]
                    achieve_name = self.id2relation[subject_data[1]]
                    game_name = self.id2name[game_id]
                    game_type_name = self.id2name[self.WOKG[game_id]["type"]]
                    answer.append(olympic_name+"中"+game_type_name+game_name+"的"+achieve_name)
        if len(answer) == 0:
            return "不知道"
        return '、'.join(answer)

    def ask_athlete_sport(self, history):
        athlete = history["athlete"]
        if athlete == '':
            return "请问是哪个运动员？"
        answer = []
        athlete_id = self.entity2id[athlete][0]
        athlete_subject_data = self.WOKG[athlete_id]["subject_data"]
        for subject_data in athlete_subject_data:
            game_id = subject_data[0]
            game_subject_data = self.WOKG[game_id]["subject_data"]
            for game_data in game_subject_data:
                o = game_data[0]
                if self.WOKG[o]["type"] == "http://xlore.org/concept3":
                    olympic_name = self.id2name[o]
                    game_name = self.id2name[game_id]
                    game_type_name = self.id2name[self.WOKG[game_id]["type"]]
                    answer.append(olympic_name + "中" + game_type_name + game_name)
        if len(answer) == 0:
            return "不知道"
        return '、'.join(answer)

    def ask_athlete_wo(self, history):
        athlete = history["athlete"]
        if athlete == '':
            return "请问是哪个运动员？"
        answer = []
        athlete_id = self.entity2id[athlete][0]
        athlete_subject_data = self.WOKG[athlete_id]["subject_data"]
        for subject_data in athlete_subject_data:
            game_id = subject_data[0]
            game_subject_data = self.WOKG[game_id]["subject_data"]
            for game_data in game_subject_data:
                o = game_data[0]
                if self.WOKG[o]["type"] == "http://xlore.org/concept3":
                    answer.append(self.id2name[o])
        if len(answer) == 0:
            return "没有参加任何冬奥会（也可能小冬没有学习相关知识）"
        return ','.join(answer)

    def ask_venue_sport(self, history):
        venue = history["venue"]
        if venue == '':
            return "请问是哪个场馆？"
        answer = []
        venue_id = self.entity2id[venue][0]
        venue_object_data = self.WOKG[venue_id]["object_data"]
        for object_data in venue_object_data:
            if object_data[1] == "http://xlore.org/olympic/property/50":
                answer.append(object_data[2])
        if len(answer) == 0:
            return "不知道"
        return ','.join(answer)

    def ask_venue_wo(self, history):
        venue = history["venue"]
        if venue == '':
            return "请问是哪个场馆？"
        answer = []
        venue_id = self.entity2id[venue][0]
        venue_subject_data = self.WOKG[venue_id]["subject_data"]
        for subject_data in venue_subject_data:
            if subject_data[1] == "http://xlore.org/olympic/property/52":
                answer.append(self.id2name[subject_data[0]])
        if len(answer) == 0:
            return "不知道"
        return ','.join(answer)





