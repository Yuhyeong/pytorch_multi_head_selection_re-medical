import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from torchcrf import CRF
from pytorch_transformers import *


# 一个网络
class MultiHeadSelection_for_test(nn.Module):

    def __init__(self, hyper) -> None:
        super(MultiHeadSelection_for_test, self).__init__()
        device = torch.device("cuda")
        # 总参数
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu
        # 加载vocab
        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r', encoding='utf-8'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r', encoding='utf-8'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r', encoding='utf-8'))
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}
        # 词向量
        # 定义一个word_embedding的存储结构
        # num_embeddings几个词向量
        # embedding_dim词向量维度
        # 从word_vocab构建字数个向量，每个向量300维
        # 只用输入个数与维数即可生成一组对应的向量
        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
            embedding_dim=hyper.emb_size)

        # 定义一个relation_emb的存储结构
        # 从word_vocab构建关系数目个向量，每个向量100维
        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
            embedding_dim=hyper.rel_emb_size)

        # bio + pad
        # 定义一个relation_emb的存储结构，同时有mask过的pad
        # 从bio_vocab构建bio数量个向量，每个向量100维
        # 生成向量
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.bio_emb_size)

        # 预训练的模型
        # chineses_selection中用的是LSTM的encoder
        # 直接用pytorch封装好的LSTM来定义pytorch，即网络结构
        # 网络的输入结构为1.每一个词的维度（字维度） 2.hidden_size 3.是否是双向的LSTM，4.输入向量的第一维是否是batch数目
        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
        elif hyper.cell_name == 'bert':
            self.post_lstm = nn.LSTM(hyper.emb_size,
                                     hyper.hidden_size,
                                     bidirectional=True,
                                     batch_first=True)
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
            for name, param in self.encoder.named_parameters():
                if '11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    # print(name, param.size())
        else:
            raise ValueError('cell name should be gru/lstm/bert!')

        # 激活函数
        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        # 随机打标签机器，CRF，条件随机场。 作用是输入一个bio标签序列。输出一个
        # CRF负责学习相邻实体标签之间的转移规则，即让b i o三个标签能尽量输出成一个实体，让字向量之间产生关系，以便升格为词向量
        # 1.标签数量 2.第一维是否是batch数目
        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)

        self.selection_u = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_v = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                      hyper.rel_emb_size)
        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)

        self.bert2hidden = nn.Linear(768, hyper.hidden_size)
        # for bert_lstm
        # self.bert2hidden = nn.Linear(768, hyper.emb_size)

        if self.hyper.cell_name == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

        # self.accuracy = F1Selection()

    def inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
            -1, -1, len(self.relation_vocab),
            -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold

        selection_triplets = self.selection_decode(text_list, decoded_tag,
                                                   selection_tags)
        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
            -1, -1, len(self.relation_vocab),
            -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['selection_loss'].item(), epoch, epoch_num)

    # sample是从dataloader中__getitem__返回的
    # 若is_train为false，则返tensor的json文件
    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:
        device = torch.device("cuda")

        # 这三个为tensor集合

        # 取text、relation、bio的对应位置在vocab里面的序号
        # tokens 二维矩阵，batch_size个max_len的向量，e.g 4个300维向量，4个最大长度为300的text向量。字向量，存的是序号word_vocab里面的字序号
        # selection 三维矩阵，存储的关系向量，300*300*48即300个实体之间的40个关系。关系向量，存的是序号relation_vocab里面的字序号
        # bio_gold，二维矩阵 与tokens字向量相对应的BIO向量同为*300,，存的是序号bio_vocab里面的BIO序号
        tokens = sample.tokens_id.to(device)  # text
        # selection_gold = sample.selection_id.to(device)  # selection
        # bio_gold = sample.bio_id.to(device)  # bio
        # tokens = sample.tokens_id
        # selection_gold = sample.selection_id
        # bio_gold = sample.bio_id

        # 取text、relation、bio的实际内容
        text_list = sample.text
        # spo_gold = sample.spo_gold
        # bio_text = sample.bio

        # 判断tokens是否填充，pad的作用是填充，是长短句的input一样，即用来断定长短句的边界
        if self.hyper.cell_name in ('gru', 'lstm'):
            # 判断token每个位置的序号是不是<pad>，即这个地方是否是填充为，若是则TRUE，e.g.输出4*300的bool元素
            mask = tokens != self.word_vocab['<pad>']  # batch x seq #取bio_vocab里面的<pad>的序号
            #通过bio_mask来判断长短句边界
            bio_mask = mask
        elif self.hyper.cell_name in ('bert'):
            notpad = tokens != self.bert_tokenizer.encode('[PAD]')[0]
            notcls = tokens != self.bert_tokenizer.encode('[CLS]')[0]
            notsep = tokens != self.bert_tokenizer.encode('[SEQ]')[0]
            mask = notpad & notcls & notsep
            bio_mask = notpad & notsep  # fst token for crf cannot be masked
        else:
            raise ValueError('unexpected encoder name!')

        if self.hyper.cell_name in ('lstm', 'gru'):
            # 输入的是每个句子text的对应的，每个字的位置的字向量，根据vocab里的字序号来从word_embeddings中提取字向量
            # 4*300*300， 4个300长度的序列，每个字都是一个300维度的字向量
            embedded = self.word_embeddings(tokens)
            o, h = self.encoder(embedded)
            o = (lambda a: sum(a) / 2)(torch.split(o,
                                                   self.hyper.hidden_size,
                                                   dim=2))
        elif self.hyper.cell_name == 'bert':
            # with torch.no_grad():
            # 输入字向量
            o = self.encoder(tokens, attention_mask=mask)[
                0]  # last hidden of BERT
            # o = self.activation(o)
            # torch.Size([16, 310, 768])
            o = self.bert2hidden(o)

            # below for bert+lstm
            # o, h = self.post_lstm(o)

            # o = (lambda a: sum(a) / 2)(torch.split(o,
            #                                        self.hyper.hidden_size,
            #                                        dim=2))
        else:
            raise ValueError('unexpected encoder name!')
        emi = self.emission(o)

        # 开始确定输出的内容
        output = {}

        crf_loss = 0

        # 若forward中的is_train设置为fale，则output会返回bio的tensor
        if is_train:
            #crf误差，用于调整参数，是标注的字向量能组成词向量
            #        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            # 根据给出的emissions评分返回，mask为输入序列的定界
            crf_loss = -self.tagger(emi, bio_gold,
                                    mask=bio_mask, reduction='mean')
        else:
            # decoded_tag为预测出来的text的实体标注序列
            #        """Find the most likely tag sequence using Viterbi algorithm.
            # 返回的是在这个crf_Loss下，可能性最高的向量序列
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            output['decoded_tag'] = [list(map(lambda x: self.id2bio[x], tags)) for tags in decoded_tag]
            # output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).to(device)
            # bio_gold = torch.tensor(temp_tag)

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        # # forward multi head selection前向多头
        # B, L, H = o.size()
        # u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        # v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        # uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # # correct one
        # selection_logits = torch.einsum('bijh,rh->birj', uv,
        #                                 self.relation_emb.weight)

        # use loop instead of matrix
        # selection_logits_list = []
        # for i in range(self.hyper.max_text_len):
        #     uvi = uv[:, i, :, :]
        #     sigmoid_input = uvi
        #     selection_logits_i = torch.einsum('bjh,rh->brj', sigmoid_input,
        #                                         self.relation_emb.weight).unsqueeze(1)
        #     selection_logits_list.append(selection_logits_i)
        # selection_logits = torch.cat(selection_logits_list,dim=1)

        # if not is_train:
        #     output['selection_triplets'] = self.inference(
        #         mask, text_list, decoded_tag, selection_logits)
        #     output['spo_gold'] = spo_gold
        #
        selection_loss = 0
        # if is_train:
        #     selection_loss = self.masked_BCEloss(mask, selection_logits,
        #                                          selection_gold)

        loss = crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)

        # 输出已标注的各类信息
        tagged_text = {}
        tagged_text["源文本"] = text_list
        tagged_text["标注BIO"] = output["decoded_tag"]
        # tagged_text["每个字的张量"] = embedded

        tagged_text_save_path = "output/"
        tagged_text_name = "tagged_text.json"
        tagged_text_name_proce = "tagged_text_pro.json"

        if not os.path.exists(tagged_text_save_path):
            os.mkdir(tagged_text_save_path)
        with open(os.path.join(tagged_text_save_path, tagged_text_name), 'a', encoding="utf-8") as f:
            f.write(json.dumps(tagged_text, ensure_ascii=False))
            f.write('\n')

        return output
        # decoded_tag = 预测标注BIO
        # gold_tags = 真实BIO
        # selection_triplets = 抽取的三元组
        # spo_gold = 真实三元组
        # loss 有三个 crf_loss, selection_loss loss = crf_loss + selection_loss

    def selection_decode(self, text_list, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)

            assert object != '' and subject != ''

            triplet = {
                'object': object,
                'predicate': predicate,
                'subject': subject
            }
            result[b].append(triplet)
        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass
