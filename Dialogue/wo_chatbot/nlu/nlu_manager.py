#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
from nlu.ner_model import BiLSTM_CRF
from nlu.intent_model import Intent_Model
from utils import normalize_text, find_entity
from torch.autograd import Variable
import pickle


class NLU_Manager(object):

    def __init__(self, config):

        self.use_gpu = False
        map_location = 'cpu'
        if config.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(int(config.er_gpu))
            self.use_gpu = True
            map_location = 'gpu:'+config.er_gpu

        # load ner model
        with open(config.ner_word_tag_path, 'rb') as f:
            ner_word_tag = pickle.load(f)
        self.tag2id = ner_word_tag['tag2id']
        self.id2tag = ner_word_tag['id2tag']
        self.ner_word2id = ner_word_tag['word2id']
        self.ner_id2word = ner_word_tag['id2word']
        self.ner_model = BiLSTM_CRF(len(self.ner_word2id), 100, 300, self.tag2id, None, 0.5, self.use_gpu)
        self.ner_model.load_state_dict(torch.load(config.ner_model_path, map_location=map_location))
        self.ner_model.eval()
        print("ner model load successful!")

        # load intent model
        with open(config.intent_word_dic_path, 'rb') as f:
            intent_word_dic = pickle.load(f)
        self.intent_word2id = intent_word_dic['word2id']
        self.intent_id2word = intent_word_dic['id2word']
        self.intent_model = Intent_Model(len(self.intent_word2id), 100, 300, 13, None, 0.5, self.use_gpu)
        self.intent_model.load_state_dict(torch.load(config.intent_model_path, map_location=map_location))
        self.intent_model.eval()
        print("intent model load successful!")




    def intent_dection(self, text):
        n_text = normalize_text(text)
        n_text_ids = [self.intent_word2id[i] if i in self.intent_word2id else self.intent_word2id['<UNK>'] for i in n_text]
        n_text_ids = Variable(torch.LongTensor(n_text_ids))
        with torch.no_grad():
            if self.use_gpu:
                socre = self.intent_model(n_text_ids.cuda())
            else:
                socre = self.intent_model(n_text_ids)
        socre = torch.nn.Softmax()(socre)
        pred = torch.argmax(socre).item()
        return pred

    def ner_tagger(self, text):

        n_text = normalize_text(text)
        n_text_ids = [self.ner_word2id[i] if i in self.ner_word2id else self.ner_word2id['<UNK>'] for i in n_text]
        n_text_ids = Variable(torch.LongTensor(n_text_ids))

        with torch.no_grad():
            if self.use_gpu:
                _, out = self.ner_model(n_text_ids.cuda())
            else:
                _, out = self.ner_model(n_text_ids)
        entitys, positions = find_entity(text, out, self.id2tag)
        return entitys, positions

    def nlu_answer(self, intent, history, kg_manager):
        answer = kg_manager.id2func[intent](history)
        return answer