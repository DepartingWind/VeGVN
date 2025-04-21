from tools.pargs import pargs
import os
import os.path as osp
import json
import random
import torch
from gensim.models import Word2Vec,word2vec
import re
import jieba
from tqdm import tqdm
args = pargs()
cwd=os.getcwd()

def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text

class Embedding():
    def __init__(self, w2v_path):
        self.w2v_path = w2v_path
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = self.make_embedding()


    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        self.embedding_matrix = []
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
        for i, word in enumerate(self.embedding.wv.key_to_index):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv.get_vector(word, norm=True))
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<UNK>")
        return self.embedding_matrix

    def sentence_word2idx(self, sen):
        sentence_idx = []
        for word in sen:
            if (word in self.word2idx.keys()):
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return sentence_idx

    def get_word_embedding(self, sen):
        sentence_idx = self.sentence_word2idx(sen)
        word_embedding = self.embedding_matrix[sentence_idx]
        return word_embedding

    def get_sentence_embedding(self, sen):
        if str(args.dataset) == "PHEME":
            sen = clean_str_cut(sen)
        elif str(args.dataset) == "WeiboOurs":
            sen = clean_str_cut4Ours(sen)
        word_embedding = self.get_word_embedding(sen)
        sen_embedding = torch.sum(word_embedding, dim=0)
        return sen_embedding

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)

def clean_str_cut4Ours(string):

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(jieba.cut(string.strip().lower(), cut_all=False))
    words = [w for w in words if w not in stopwords]
    return words

def clean_str_cut(string):
    string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(string.strip().lower().split())
    return words

def collect_label_sentences(path):
    sentences = []
    for lines in tqdm(open(path)):
        line = lines.rstrip()

        if str(args.dataset) == "PHEME":
            line = clean_str_cut(line)
        elif str(args.dataset) == "WeiboOurs":
            line = clean_str_cut4Ours(line)
        sentences.append(line)
    return sentences

if str(args.dataset) == "WeiboOurs":
    jieba.setLogLevel(jieba.logging.INFO)
    jieba.set_dictionary(os.path.join('..', '..', 'data', args.dataset, 'dict.txt.big'))
    stopwords_path = os.path.join('..', '..', 'data', args.dataset, 'stopwords.txt')
    stopwords = []
    with open(stopwords_path, 'r') as f:
        for line in f.readlines():
            stopwords.append(line.strip())

def train_word2vec():

    print('Loading sentences...')
    sentences = collect_label_sentences(os.path.join('..','..','data',args.dataset,args.dataset_text+'.txt'))
    print('Training Word2Vec...')
    model = Word2Vec(sentences, vector_size=args.vector_size, window=5, min_count=1, workers=12, epochs=10, sg=1)
    return model