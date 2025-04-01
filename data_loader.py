import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#将连续的距离值转换为离散的特征
dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1  #将距离值为1的元素映射到索引1。
dis2idx[2:] = 2  #将距离值从2到数组末尾的所有元素映射到索引2。
dis2idx[4:] = 3  #将距离值从4到数组末尾的所有元素映射到索引3。
dis2idx[8:] = 4  #将距离值从8到数组末尾的所有元素映射到索引4。
dis2idx[16:] = 5  #将距离值从16到数组末尾的所有元素映射到索引5。
dis2idx[32:] = 6  #将距离值从32到数组末尾的所有元素映射到索引6。
dis2idx[64:] = 7  #将距离值从64到数组末尾的所有元素映射到索引7。
dis2idx[128:] = 8  #将距离值从128到数组末尾的所有元素映射到索引8。
dis2idx[256:] = 9  #将距离值从256到数组末尾的所有元素映射到索引9。


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))
    #确保所有句子的BERT输入具有相同的长度
    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])  #当前批次中所有句子的BERT输入的最大片段数
    bert_inputs = pad_sequence(bert_inputs, True)  #对 bert_inputs 列表中的所有BERT输入进行填充，
    # 使得它们的长度都等于最大片段数 max_pie。True 参数表示在序列的左侧进行填充。
    batch_size = bert_inputs.size(0)  #当前批次的大小

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
            torch.LongTensor(self.grid_labels[item]), \
            torch.LongTensor(self.grid_mask2d[item]), \
            torch.LongTensor(self.pieces2word[item]), \
            torch.LongTensor(self.dist_inputs[item]), \
            self.sent_length[item], \
            self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        # print(instance['sentence'])
        # print(len(instance['sentence']))
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [
            tokenizer.sep_token_id])  #tokenizer.cls_token_id表示句子的开始，tokenizer.sep_token_id表示句子的结束

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int64)  #全为0的二维数组
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool_)  #全为false的二维数组
        _dist_inputs = np.zeros((length, length), dtype=np.int64)  #全为0的二维数组
        _grid_mask2d = np.ones((length, length), dtype=np.bool_)  #全为True的二维数组，用于在处理句子或序列数据时指示哪些位置是有效的
        #_pieces2word记录每个分词（piece）在原始句子中属于哪个单词
        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1  #加1是因为在分词的前后分别加了cls和sep
                start += len(pieces)  #移动到下一个单词的开始
        #dist_inputs 矩阵中的每个元素 _dist_inputs[i, j] 表示第 i 个单词到第 j 个单词之间的距离，为负数说明j在i的后面
        for k in range(length):
            _dist_inputs[k, :] += k  #行
            _dist_inputs[:, k] -= k  #列
        #将_dist_inputs中的每个距离值转换为一个索引值
        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9  #不懂为什么要+9？？？？？？？
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19  #不懂为啥要等于19？？？？？？

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])  #为每个命名实体生成一个唯一的文本表示

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    with open('./data/{}/converted_short_train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/converted_short_val.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/converted_short_test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 指定本地模型的路径
    model_path = r".\bert_base_chinese"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)
    #创建一个表格
    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)
