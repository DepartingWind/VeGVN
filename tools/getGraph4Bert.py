import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os,sys
sys.path.append(os.getcwd())
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from MyTools.pargs import pargs
args = pargs()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from transformers import BertTokenizer, BertModel,RobertaTokenizer,RobertaModel
from tools.BertEmbed import getSentEmb
cwd=os.getcwd()

import torch
from model import longclip

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        text = tree[j]['vec']
        nodeC.word = text
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            root = nodeC
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word

    rootfeat= root_word
    matrix=np.zeros([len(index2node),len(index2node)])
    raw=[]
    col=[]
    x_text = []
    edgematrix=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                raw.append(index_i)
                col.append(index_j)
        x_text.append(index2node[index_i + 1].word)
    edgematrix.append(raw)
    edgematrix.append(col)
    return x_text, edgematrix,rootfeat,rootindex

def main():
    args = pargs()
    dataset = args.dataset
    dataset_tree = args.dataset_tree
    dataset_label = args.dataset_label
    bert_pre = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_pre)
    bert_model = BertModel.from_pretrained(bert_pre, output_hidden_states=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

    treePath = os.path.join('..','data',dataset,dataset_tree+'.txt')
    print(f"reading {dataset} tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        try:
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[3]
        except:
            print(line)
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    print('tree no:', len(treeDic))

    labelPath = os.path.join('..','data',dataset,dataset_label+'.txt')
    print(f"loading {dataset} label:")
    event,y= [],[]
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid,label = line.split(' ')[0], line.split(' ')[1]
        labelDic[eid] = int(label)
        y.append(labelDic[eid])
        event.append(eid)
        if labelDic[eid]==0:
            l1 += 1
        if labelDic[eid]==1:
            l2 += 1

    print('len(labelDic):{},len(event):{},len(y):{}'.format(len(labelDic),len(event),len(y)))
    print("non-rumours: {}, false-rumours: {}".format(l1, l2) )

    if not os.path.exists(os.path.join('..','data',f'{dataset}graph_longclip')):
        os.mkdir(os.path.join('..','data',f'{dataset}graph_longclip'))

    def combine_features(bert_embedding, clip_embedding):
        combined_embedding = torch.cat((bert_embedding, clip_embedding), dim=1)
        return combined_embedding

    def process_long_text(text, model, device, window_size=77, stride=39):
        segments = []
        start = 0
        while start < len(text):
            end = min(start + window_size, len(text))
            segments.append(text[start:end])
            start += stride

        segment_features = []

        for segment in segments:
            segment_tokens = longclip.tokenize([segment]).to(device)
            with torch.no_grad():
                segment_features.append(model.encode_text(segment_tokens).float())

        if segment_features:
            aggregated_features = torch.mean(torch.stack(segment_features), dim=0)
        else:
            raise ValueError("No valid segments could be processed.")

        return aggregated_features

    def loadEid(event,id,y):
        if not os.path.exists(os.path.join(f'../data/{dataset}graph_longclip/'+id+'.npz')):
            if event is None:
                return None
            if len(event) < 2:
                return None
            if len(event)>1:
                try:
                    x_text, tree, rootfeat, rootindex = constructMat(event)
                    x_f=[]
                    for text in x_text:
                        text_embed = getSentEmb(text,tokenizer,bert_model,'cuda:0')
                        clip_embed = process_long_text(text, model_clip, device)
                        combined_embed = combine_features(text_embed, clip_embed)

                        x_f.append(combined_embed.cpu().detach().numpy())
                    rootfeat = getSentEmb(rootfeat,tokenizer,bert_model,'cuda:0').cpu().detach().numpy()
                    rootfeat, tree, x_text, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_f), np.array(
                        rootindex), np.array(y)
                    np.savez(os.path.join(f'../data/{dataset}graph_longclip/'+id+'.npz'), x=x_text,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
                    return None
                except Exception as e:
                    print(e)

    print(f"loading {dataset} dataset ... ...", )
    for eid in tqdm(event):
        loadEid(treeDic[eid] if eid in treeDic else None, eid, labelDic[eid])
    return

if __name__ == '__main__':
    main()
