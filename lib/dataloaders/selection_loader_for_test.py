import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from pytorch_transformers import *


class Selection_Dataset_for_test(Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper
        self.data_root = hyper.data_root

        #加载processing过的train，evaluation文件
        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r',encoding='utf-8'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r',encoding='utf-8'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r',encoding='utf-8'))

        # self.selection_list = []
        self.text_list = []
        # self.bio_list = []
        # self.spo_list = []

        # for bert only
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        #按行读取train/evaluation的processing过的文件
        for line in open(os.path.join(self.data_root, dataset), 'r',encoding='utf-8'):
            line = line.strip("\n")
            instance = json.loads(line)

            #把train/evaluation的processing过的文件读取，读取其中的selection、text、bio、spo_list为局部变量list
            # self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            # self.bio_list.append(instance['bio'])
            # self.spo_list.append(instance['spo_list'])

    def __getitem__(self, index):
        #获取index位置的selection、text、bio、spo_list局部变量list的单个对应数据
        # selection = self.selection_list[index]
        text = self.text_list[index]
        # bio = self.bio_list[index]
        # spo = self.spo_list[index]

        #若为bert模式
        if self.hyper.cell_name == 'bert':
            text, bio = self.pad_bert(text, bio, selection)#读取为bert所需的形式
            tokens_id = torch.tensor(
                self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2tensor(text)#获取text对应的tensor集合

        # bio_id = self.bio2tensor(bio)#获取bio对应的tensor集合
        # selection_id = self.selection2tensor(text, selection)#获取selection对应的tensor集合

        #返回三个tensor集合, 和单个词汇
        return tokens_id, len(text), text

    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text: List[str], bio: List[str], selection: List[Dict[str, int]]) -> Tuple[List[str], List[str], Dict[str, int]]:
        # for [CLS] and [SEP]
        text = ['[CLS]'] + [text] + ['[SEP]']
        bio = ['O'] + bio + ['O']
        # selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] +
        #               1, 'predicate': triplet['predicate']} for triplet in selection]
        assert len(text) <= self.hyper.max_text_len
        text = text + ['[PAD]'] * (self.hyper.max_text_len - len(text))
        return text, bio

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer

        #oov，其全称是Out-Of-Vocabulary
        #如果出现在test数据集中的词没有出现在train中，那么这就是一个oov
        oov = self.word_vocab['oov']#获取oov的编号
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        padded_list.extend([self.word_vocab['<pad>']] *
                           (self.hyper.max_text_len - len(text)))

        #获取text中的所有词汇的tensor集合
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        # 开始mask
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] *
                           (self.hyper.max_text_len - len(bio)))

        #返回的是所有被掩盖mask的词汇的列表
        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros(
            (self.hyper.max_text_len, len(self.relation_vocab),
             self.hyper.max_text_len))
        NA = self.relation_vocab['N']
        result[:, NA, :] = 1
        for triplet in selection:

            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']

            result[subject, predicate, object] = 1
            result[subject, NA, object] = 0

        return result


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, selection_id, spo, text, bio

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        # self.bio_id = pad_sequence(transposed_data[1], batch_first=True)
        # self.selection_id = torch.stack(transposed_data[2], 0)
        #
        # self.length = transposed_data[3]
        #
        # self.spo_gold = transposed_data[4]
        self.text = transposed_data[2]
        # self.bio = transposed_data[6]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        # self.bio_id = self.bio_id.pin_memory()
        # self.selection_id = self.selection_id.pin_memory()
        return self


#非类里面的函数
def collate_fn(batch):
    return Batch_reader(batch)

# 相当于自定义一个dataloader，partial用来对函数设置默认参数
# Selection_loader相当于一个Datloader(collate_fn=collate_fn, pin_memory=True)
Selection_loader_for_test = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
