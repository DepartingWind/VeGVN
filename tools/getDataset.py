import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from PIL import Image
from torchvision import transforms
from tools.pargs import pargs
from torchvision.models import Swin_S_Weights
from transformers import BertTokenizer, BertModel
from tools.BertEmbed import getSentEmb
from tools.dct_trans import tform_freq
from tools.noise_trans import SRM
import matplotlib.pyplot as plt
from tools.ela_trans import ELA
from tools.noise_srm_new import SRM_Trans
args = pargs()
cwd = os.getcwd()

test_tfm = transforms.Compose([
    transforms.Resize(size=(args.image_size,args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

swins_tfm = Swin_S_Weights.IMAGENET1K_V1.transforms

train_tfm = transforms.Compose([
            transforms.Resize(246),
            transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def loadTree():
    treePath = os.path.join(f'../../data/{args.dataset}/{args.dataset_tree}.txt')
    print("reading Weibo tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    print('tree no:', len(treeDic))
    return treeDic

def loadBiData(fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    treeDic = loadTree()
    data_path = os.path.join('..','..','data', args.dataset + args.Gpath)
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path,tfm=test_tfm)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path,tfm=test_tfm)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', f'{args.dataset}graph'),tfm = test_tfm):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.transform = tfm

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        try:
            data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        except:
            print("id:",id)

        try:
            edgeindex = data['edgeindex']
            if self.tddroprate > 0:
                row = list(edgeindex[0])
                col = list(edgeindex[1])
                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
                poslist = sorted(poslist)
                row = list(np.array(row)[poslist])
                col = list(np.array(col)[poslist])
                new_edgeindex = [row, col]
            else:
                new_edgeindex = edgeindex
        except:
            print(f'id:{id}')

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]

        try:
            if os.path.exists(f'../../data/{args.dataset}_pic/'+id+'.jpg'):
                im = Image.open(f'../../data/{args.dataset}_pic/'+id+'.jpg')
            if os.path.exists(f'../../data/{args.dataset}_pic/'+id+'.png'):
                im = Image.open(f'../../data/{args.dataset}_pic/'+id+'.png')
            im = im.convert("RGB")
            im = self.transform(im)
        except Exception as e:
            print(e,' and id is:',id)
        return Data(x=torch.tensor(data['x'], dtype=torch.float32).squeeze(1),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),img=im.unsqueeze(0),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def loadUdData(fold_x_train, fold_x_test, droprate):
    treeDic = loadTree()
    data_path = os.path.join('..','..','data', args.dataset + args.Gpath)
    print("loading train UDset", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, data_path=data_path,tfm=test_tfm)
    print("train no:", len(traindata_list))
    print("loading test UDset", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic, data_path=data_path,tfm=test_tfm)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000,
                 data_path=os.path.join('..','..','data', 'Weibograph'), clip_data_path=os.path.join('..','..','data', f'{args.dataset}graph_clip'), tfm = test_tfm):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.clip_data_path = clip_data_path
        self.transform = tfm
        self.transform_freq = tform_freq

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        clip_data = np.load(os.path.join(self.clip_data_path, id + ".npz"), allow_pickle=True)

        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        new_edgeindex = [row, col]

        try:
            if os.path.exists(f'../../data/{args.dataset}_pic/' + id + '.jpg'):
                im = Image.open(f'../../data/{args.dataset}_pic/' + id + '.jpg')
            else:
                print(f'../../data/{args.dataset}_pic/' + id + '.jpg does not EXIST!!!')
            init_im = im
            im = im.convert("RGB")

            im = self.transform(im)
            im_noise = np.asarray(im.permute(1,2,0))
            im_noise = SRM_Trans(im_noise)
            my_Data = Data(x=torch.tensor(data['x'], dtype=torch.float32).squeeze(1),
                           clip_x=torch.tensor(clip_data['x'], dtype=torch.float32).squeeze(1),
                           edge_index=torch.LongTensor(new_edgeindex),
                           y=torch.LongTensor([int(data['y'])]),
                           rootindex=torch.LongTensor([int(data['rootindex'])]), init_im=init_im, img=im.unsqueeze(0),
                           img_noise = torch.tensor(im_noise, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                           )
        except Exception as e:
            print(e, ' and id is:', id)
        return my_Data