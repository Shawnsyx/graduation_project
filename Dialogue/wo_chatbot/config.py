# -*- coding: utf-8 -*-


import argparse


def make_config():

    parser = argparse.ArgumentParser()


    # data path
    parser.add_argument('--id2entity', default='./data/id2entity.txt')
    parser.add_argument('--id2name', default='./data/id2name.txt')
    parser.add_argument('--id2relation', default='./data/id2relation.txt')
    parser.add_argument('--id2type', default='./data/id2type.txt')
    parser.add_argument('--sport2game', default='./data/sport2game.txt')
    parser.add_argument('--WOKG', default='./data/WOKG.txt')

    parser.add_argument('--ner_model_path', default='./data/ner_model.ckpt')
    parser.add_argument('--ner_word_tag_path', default='./data/ner_word_tag.pkl')

    parser.add_argument('--intent_model_path', default='./data/intent_model.ckpt')
    parser.add_argument('--intent_word_dic_path', default='./data/intent_word_dic.pkl')

    parser.add_argument('--gpu', default='0')



    return parser.parse_args()

