#!/usr/bin/python
# -*- coding:utf8 -*-

from config import make_config
from nlu.nlu_manager import NLU_Manager
from nlu.kg_manager import KG_Manager
class Dailog(object):

    def __init__(self, config):
        self.config = config
        self.nlu_mananger = NLU_Manager(config)
        self.kg_manager = KG_Manager(config)

        self.history = {
            "olympic": "",
            "sport": "",
            "game": "",
            "athlete": "",
            "venue": ""
        }

        self.start_str = ["hello", "你好", "您好"]
        self.end_str = ["再见", "拜拜", "byebye"]

    def answer(self, text):
        # self.kg_manager.clear_entity2id()
        if text in self.start_str:
            return "你好"
        if text in self.end_str:
            return "再见！"

        intent = self.nlu_mananger.intent_dection(text)
        print("intent:", intent)
        entities, positions = self.nlu_mananger.ner_tagger(text)
        print("slots:", entities, positions)
        for entity in entities:
            entity_type, entity_name = self.kg_manager.entity2type(entity)
            # print(entity, entity_type, entity_name)
            if entity_name == "普莱西德湖冬奥会":
                return "普莱西德湖一共举办过两届冬奥会，分别为第三届冬奥会和第十三届冬奥会，您指的是哪一个？"
            if entity_name == "圣莫里茨冬奥会":
                return "圣莫里茨冬一共举办过两届冬奥会，分别为第二届冬奥会和第五届冬奥会，您指的是哪一个？"
            if entity_name == "因斯布鲁克冬奥会":
                return "因斯布鲁克一共举办过两届冬奥会，分别为第九届冬奥会和第十二届冬奥会，您指的是哪一个？"
            if entity_type != None:
                self.history[entity_type] = entity_name
        print(self.history)
        return self.nlu_mananger.nlu_answer(intent, self.history, self.kg_manager)


if __name__ == '__main__':
    config = make_config()
    dialog = Dailog(config)
    # print(dialog.answer("2010年冬季奥林匹克运动会的吉祥物是什么？")) # 0
    # print(dialog.answer("第二十一届冬奥会有哪些参赛国家？")) #1
    # print(dialog.answer("第二十一届冬奥会有哪些比赛场馆？")) # 2
    # print(dialog.answer("第二十一届冬季奥林匹克运动会有哪些比赛项目？")) # 3
    # print(dialog.answer("钢架雪车有哪些子项目？"))  # 4
    # print(dialog.answer("第二十三届冬奥会中速度滑冰女子500米的第一名是谁？")) # 5
    # print(dialog.answer("第二十三届冬奥会中速度滑冰女子500米的第二名是谁？")) # 6
    # print(dialog.answer("第二十三届冬奥会中速度滑冰女子500米的第三名是谁？")) # 7
    # print(dialog.answer("武大靖取得了哪些成绩？")) # 8
    # print(dialog.answer("武大靖参加了哪些比赛？"))  # 9
    # print(dialog.answer("武大靖参加了哪几届冬奥会？")) # 10
    # print(dialog.answer("龙平滑雪渡假村举办过哪些比赛？")) # 11
    # print(dialog.answer("龙平滑雪渡假村举办过第几届冬奥会？")) # 12
    while True:
        text = input("用户：")
        answer = dialog.answer(text)
        print("bot：", answer)