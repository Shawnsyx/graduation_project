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
    # parser.add_argument('--kb_path', default='./data/kb_data.pkl')
    parser.add_argument('--word_tag_path', default='./data/word_tag.pkl')
    parser.add_argument('--model_path', default='./data/')

    parser.add_argument('--er_gpu', default='4')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--er_epochs', default=5, type=int, help='epochs')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.015, type=float)

    parser.add_argument('--earlystop', default=100, type=int)
    parser.add_argument('--delta', default=0, type=float)


    # Network
    parser.add_argument('--num_layers', default=2, type=int, help='lstm layers')
    parser.add_argument('--hidden_size', default=1024, type=int, help='lstm hidden size')
    parser.add_argument('--embed_size', default=300, help='word embedding size')
    parser.add_argument('--use_pre_embedding', default=True, type=bool)

    return parser.parse_args()

