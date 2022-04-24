import os
import json
import time
import argparse

import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD
from pytorch_transformers import AdamW, WarmupLinearSchedule

from lib.preprocessings import Chinese_selection_preprocessing, Conll_selection_preprocessing, Conll_bert_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.metrics import F1_triplet, F1_ner
from lib.models import MultiHeadSelection
from lib.models import M
from lib.config import Hyper

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='chinese_selection_re',
                    help='experiments/exp_name.json     (chinese_selection_re, chinese_bert_re, conll_re, conll_bert_re)')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='train',
                    help='preprocessing|train|evaluation|test')
args = parser.parse_args()


class Runner(object):
    # 初始化
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'

        self.hyper = Hyper(os.path.join('experiments',
                                        self.exp_name + '.json'))

        self.gpu = self.hyper.gpu
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None

    # 优化器
    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5),
            'adamw': AdamW(model.parameters())
        }
        return m[name]

    # 初始化模型，在selection中
    def _init_model(self):
        device = torch.device("cuda")

        #定义一个网络，在selection中
        #使用hyper中的参数进行训练
        self.model = MultiHeadSelection(self.hyper).to(device)
        # self.model = MultiHeadSelection(self.hyper)

    # test的model
    def _init_model_of_test(self):
        self.model = MultiHeadSelection(self.hyper)

    # 预训练
    def preprocessing(self):
        if self.exp_name == 'conll_selection_re':
            self.preprocessor = Conll_selection_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_selection_re':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        elif self.exp_name == 'conll_bert_re':
            self.preprocessor = Conll_bert_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_bert_re':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)   #输入epoch号，选择使用哪个epoch的模型
            self.evaluation()
        elif mode == 'test':
            self._init_model()
            self.load_model(epoch=self.hyper.test_epoch)
            self.test()
        else:
            raise ValueError('invalid mode')

    # 加载模型
    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             self.exp_name + '_' + str(epoch))))

    # 存储模型
    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    # 评估
    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            print('Triplets-> ' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ]))

    # 测试
    def test(self):
        test_set = Selection_Dataset(self.hyper, self.hyper.test)#dataset
        loader = Selection_loader(test_set, batch_size=self.hyper.test_batch, pin_memory=True)#dataloader

        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))#进度条

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)

    # 训练
    def train(self):

        #数据集和dataloader
        train_set = Selection_Dataset(self.hyper, self.hyper.train)

        #返回三个tensor集合, 和单个词汇
        #__getitem__ 会  return tokens_id, bio_id, selection_id, len(text), spo, text, bio
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=True)
        device = torch.device("cuda")

        #按epoch训练
        for epoch in range(self.hyper.epoch_num):
            self.model.train()#开始训练按照selection文件夹中定义的网络，本质是一个nn.modules

            #进度条
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))
            #按batch训练
            for batch_idx, sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)#决定 is_train
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.save_model(epoch)#存现在调过参数的模型

            if epoch % self.hyper.print_epoch == 0 and epoch > 3:#每3次evaluation验证一次
                self.evaluation()


if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)
