# -*- coding: utf-8 -*-

"""

@author: cdtang
@file: config.py
@time: 19-11-12 下午9:03

"""

import argparse


def make_config():

    parser = argparse.ArgumentParser()


    # train
    parser.add_argument('--train_data_path', default='./data/train_data.pkl')
    parser.add_argument('--dev_data_path', default='./data/dev_data.pkl')
    parser.add_argument('--test_data_path', default='./data/test_data.pkl')
    parser.add_argument('--kb_path', default='./data/kb_data.pkl')
    parser.add_argument('--word_tag_path', default='./data/word_dic.pkl')
    parser.add_argument('--model_path', default='./data/')

    parser.add_argument('--ed_gpu', default='1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--ed_epochs', default=5, type=int, help='epochs')

    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)

    parser.add_argument('--earlystop', default=100, type=int)
    parser.add_argument('--delta', default=0, type=float)


    # Network
    parser.add_argument('--hidden_size', default=1024, type=int, help='lstm hidden size')
    parser.add_argument('--embed_size', default=300, help='word embedding size')
    parser.add_argument('--use_pre_embedding', default=True, type=bool)

    return parser.parse_args()

