import datetime
import time
import sys, os

sys.path.append(os.getcwd())
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from tools.pargs import pargs

args = pargs()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from tools.getDataset import loadUdData
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from tools.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv, GAT
from torchvision import models
import torch.nn.init as init
from PIL import Image
from torchvision import transforms
import copy
from torchvision.models import Swin_B_Weights, VGG19_Weights, Swin_S_Weights, Swin_T_Weights, ResNet50_Weights
from Fusion.co_attention import BertConfig, BertConnectionLayer
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn as nn

cwd = os.getcwd()

myseed = args.seed
np.random.seed(myseed)
random.seed(myseed)
torch.cuda.set_device(0)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

from process.model import longclip
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = longclip.load("./../checkpoints/longclip-B.pt", device=device)

from transformers import CLIPProcessor
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

class FE_GCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(FE_GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.co_att = CoSelfAttention(hidden_size=out_feats,
                                      num_attention_heads=args.bi_num_attention_heads,
                                      bi_hidden_size=out_feats,
                                      attention_probs_dropout_prob=args.attention_probs_dropout_prob)
        self.text_fc = torch.nn.Linear(1280, args.vector_size)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.text_fc(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        x = F.relu(x)

        rootindex = data.rootindex
        batch_size = max(data.batch) + 1
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]

        x, r = self.co_att(x, root_extend)
        x = torch.cat((x, r), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x


class FE_ClipGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(FE_ClipGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.co_att = CoSelfAttention(hidden_size=out_feats,
                                      num_attention_heads=args.bi_num_attention_heads,
                                      bi_hidden_size=out_feats,
                                      attention_probs_dropout_prob=args.attention_probs_dropout_prob)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.clip_x, data.edge_index

        x = self.drop(x)
        x = self.relu(x)
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        x = F.relu(x)

        rootindex = data.rootindex
        batch_size = max(data.batch) + 1
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]

        x, r = self.co_att(x, root_extend)
        x = torch.cat((x, r), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class TransformerBlock(torch.nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = torch.nn.LayerNorm(normalized_shape=input_size)

        self.W_q = torch.nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = torch.nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = torch.nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = torch.nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = torch.nn.Linear(input_size, input_size)
        self.linear2 = torch.nn.Linear(input_size, input_size)

        self.dropout = torch.nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class VisFeat(torch.nn.Module):
    def __init__(self, hid_feats, out_feats):
        super(VisFeat, self).__init__()
        we = Swin_T_Weights.IMAGENET1K_V1
        self.v_encoding = models.swin_t(weights=we)
        self.v_encoding.head = torch.nn.Linear(768, hid_feats)
        torch.nn.init.eye_(self.v_encoding.head.weight)
        self.relu = torch.nn.LeakyReLU()

        self.noise_encoding = models.swin_t(weights=we)
        self.noise_encoding.head = torch.nn.Linear(768, hid_feats)
        torch.nn.init.eye_(self.noise_encoding.head.weight)

        self.im_size = out_feats + hid_feats
        self.self_atten = TransformerBlock(input_size=(out_feats + hid_feats), n_heads=8, attn_dropout=0)
        self.dp = torch.nn.Dropout(0.3)

    def forward(self, data):
        im = self.v_encoding(data.img)
        im = self.relu(im)
        im = self.dp(im)
        im_noise = self.noise_encoding(data.img_noise)
        im_noise = self.relu(im_noise)
        im_noise = self.dp(im_noise)
        im = torch.cat((im, im_noise), 1)

        bsz = im.size()[0]

        im = self.self_atten(im.view(bsz, -1, self.im_size), im.view(bsz, -1, self.im_size),
                             im.view(bsz, -1, self.im_size))
        im = self.relu(im)

        return im.squeeze(1)

class CoSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size=512, num_attention_heads=8, bi_hidden_size=512, attention_probs_dropout_prob=0.1):
        super(CoSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.bi_hidden_size = bi_hidden_size
        self.attention_head_size = int(self.bi_hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.query1 = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.key1 = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.value1 = torch.nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout1 = torch.nn.Dropout(self.attention_probs_dropout_prob)

        self.query2 = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.key2 = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.value2 = torch.nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout2 = torch.nn.Dropout(self.attention_probs_dropout_prob)

        self.liner_o1 = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.liner_o2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout3 = torch.nn.Dropout(0.3)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, input_tensor2, attention_mask1=None, attention_mask2=None, co_attention_mask=None,
                use_co_attention_mask=False, ):
        input_tensor1 = input_tensor1.unsqueeze(1)
        input_tensor2 = input_tensor2.unsqueeze(1)

        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)

        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_probs1 = torch.nn.Softmax(dim=-1)(attention_scores1)

        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1).squeeze(1)

        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_probs2 = torch.nn.Softmax(dim=-1)(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2).squeeze(1)

        context_layer1 = self.dropout3(self.liner_o1(context_layer1))
        context_layer2 = self.dropout3(self.liner_o2(context_layer2))

        return context_layer1, context_layer2


class Net(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.single_size = (out_feats + hid_feats) * 2

        self.FE_GCN = FE_GCN(in_feats, hid_feats, out_feats)
        self.vis_extractor = VisFeat(hid_feats, out_feats)

        self.FE_ClipGCN = FE_ClipGCN(512, hid_feats, out_feats)
        self.clip_img_encoder = nn.Sequential(
            nn.Linear(512, hid_feats),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.clip_gcn_proj = nn.Linear((hid_feats + out_feats), hid_feats)
        self.clip_img_proj = nn.Linear(hid_feats, hid_feats)
        self.clip_co_attention = CoSelfAttention(
            hidden_size=hid_feats,
            num_attention_heads=args.bi_num_attention_heads,
            bi_hidden_size=hid_feats,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob
        )
        self.clip_fused_linear = nn.Linear(2 * hid_feats, hid_feats + out_feats)

        self.fc0 = torch.nn.Linear((out_feats + hid_feats)*2 + 512*2, (out_feats + hid_feats)*2)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, 2)

        self.dropout = torch.nn.Dropout(args.dropout)
        self.relu = torch.nn.ReLU()

        self.shared_linear = torch.nn.Linear((out_feats + hid_feats), (out_feats + hid_feats))
        self.clip_shared_linear = torch.nn.Linear((out_feats + hid_feats), (out_feats + hid_feats))
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.sigm = torch.nn.Sigmoid()

        self.clip_sim_weight = 0.3

        self.co_attention = CoSelfAttention(hidden_size=(out_feats + hid_feats),
                                            num_attention_heads=args.bi_num_attention_heads,
                                            bi_hidden_size=(out_feats + hid_feats),
                                            attention_probs_dropout_prob=args.attention_probs_dropout_prob)

        self.hid_feats = hid_feats

    def Fusion(self, data):
        x = self.FE_GCN(data)
        im = self.vis_extractor(data)
        im = self.shared_linear(im)

        clip_gcn = self.FE_ClipGCN(data)
        clip_gcn = self.clip_shared_linear(clip_gcn)

        device1 = next(self.parameters()).device
        clip_feats = []
        for im1 in data.init_im:
            if isinstance(im1, torch.Tensor):
                im1 = im1.to(device1)
                im1 = transforms.ToPILImage()(im1)
            clip_feat = preprocess(im1)
            clip_feat = clip_feat.to(device1)
            clip_feats.append(clip_feat)
        clip_feats = torch.stack(clip_feats)
        clip_img_feat = self.clip_img_encoder(clip_feats)

        clip_sim = self.cos(clip_gcn, clip_img_feat)
        clip_sim = self.sigm(clip_sim)

        clip_gcn_proj = self.clip_gcn_proj(clip_gcn)
        clip_img_proj = self.clip_img_proj(clip_img_feat)
        clip_co, img_co = self.clip_co_attention(clip_gcn_proj, clip_img_proj)
        clip_fused = torch.cat((clip_co, img_co), dim=1)
        clip_fused = self.clip_fused_linear(clip_fused)

        fused_im = torch.cat((clip_feats, im), dim=1)
        if fused_im.size(-1) != (out_feats + hid_feats):
            fused_linear_layer = nn.Linear(fused_im.size(-1), out_feats + hid_feats).to(device1)
            fused_im = fused_linear_layer(fused_im)
        fused_im = self.shared_linear(fused_im)

        x = self.shared_linear(x)

        co_img, co_g = self.co_attention(fused_im, x)
        attention_fused = torch.cat((co_img, co_g), 1)
        final_fused = torch.cat([attention_fused, clip_fused], dim=1)

        out_cos = self.cos(x, fused_im)
        out_cos = self.sigm(out_cos)
        total_sim = out_cos + self.clip_sim_weight * clip_sim

        final_f = self.fc0(final_fused)
        final_f = self.relu(final_f)
        final_f = self.dropout(final_f)
        final_f = self.fc(final_f)
        final_f = F.log_softmax(final_f, dim=1)
        return final_f, total_sim


    def forward(self, data):
        x, out_cos = self.Fusion(data)
        return x, out_cos


def train_VGA(x_test, x_train, lr, weight_decay, patience, n_epochs, batchsize, iter):
    model = Net(args.vector_size, hid_feats, out_feats).to(device)
    if args.diff_lr and args.is_vision_graph:
        Diff_params = list(map(id, model.FE_GCN.conv1.parameters()))
        base_params = filter(lambda p: id(p) not in Diff_params, model.parameters())
        if args.dataset == "DRWeibo":
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.FE_GCN.conv1.parameters(), 'lr': lr}
            ], lr=lr, weight_decay=weight_decay)
        elif args.dataset == "WeiboCED":
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.FE_GCN.conv1.parameters(), 'lr': lr / 2}
            ], lr=lr, weight_decay=weight_decay)
        elif args.dataset == "Twitter":
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.FE_GCN.conv1.parameters(), 'lr': lr / 2}
            ], lr=lr, weight_decay=weight_decay)
        elif args.dataset == "PHEME":
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.FE_GCN.conv1.parameters(), 'lr': lr / 2}
            ], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, min_lr=0.0001)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadUdData(x_train, x_test, args.droprate)
    train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=False, num_workers=4)
    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=4)

    for epoch in range(n_epochs):
        avg_loss, avg_acc = [], []
        batch_idx = 0
        model.train()
        tqdm_train_loader = tqdm(train_loader)
        Train_Loss, Train_Accuracy = [], []
        for Batch_data in tqdm_train_loader:
            Batch_data = Batch_data.to(device)

            out_labels, out_cos = model(Batch_data)
            out_cos = 1 - out_cos
            loss_y = F.nll_loss(out_labels, Batch_data.y)
            loss_cos = F.binary_cross_entropy(out_cos, torch.tensor(Batch_data.y, dtype=torch.float32))
            loss = args.alpha * loss_y + (1 - args.alpha) * loss_cos
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            Train_Loss.append(loss.item())
            Train_Accuracy.append(train_acc)
            postfix = "Iter {:03d} | lr {} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Batch_Acc {:.4f}| Total_Train_Accuracy {:.4f}".format(
                iter,
                scheduler.optimizer.param_groups[0]['lr'],
                epoch,
                batch_idx,
                loss.item(),
                train_acc,
                np.mean(Train_Accuracy))
            tqdm_train_loader.set_postfix_str(postfix)
            batch_idx = batch_idx + 1
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses, temp_val_accs, temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data = Batch_data.to(device)
            val_out, val_cos = model(Batch_data)
            val_cos = 1 - val_cos
            loss_y = F.nll_loss(val_out, Batch_data.y)
            loss_cos = F.binary_cross_entropy(val_cos, torch.tensor(Batch_data.y, dtype=torch.float32))
            val_loss = args.alpha * loss_y + (1-args.alpha) * loss_cos

            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        scheduler.step(np.mean(temp_val_losses))
        print("Epoch {:05d} | lr {}| Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch,
                                                                                  scheduler.optimizer.param_groups[0][
                                                                                      'lr'],
                                                                                  np.mean(temp_val_losses),
                                                                                  np.mean(temp_val_accs)))
        with open(log_file, 'a+', encoding='UTF-8') as fp:
            fp.write("Epoch {:05d} | lr {}| Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch,
                                                                                         scheduler.optimizer.param_groups[
                                                                                             0]['lr'],
                                                                                         np.mean(temp_val_losses),
                                                                                         np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        print('results:', res)
        with open(log_file, 'a+', encoding='UTF-8') as fp:
            fp.write('\nacc:{:.4f}\n'.format(np.mean(temp_val_Acc_all)))
            fp.write('C1:{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                               np.mean(temp_val_Recll1), np.mean(temp_val_F1)))
            fp.write('C2:{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                               np.mean(temp_val_Recll2), np.mean(temp_val_F2)))

        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2), np.mean(temp_val_Prec1),
                       np.mean(temp_val_Prec2), np.mean(temp_val_Recll1), np.mean(temp_val_Recll2),
                       np.mean(temp_val_F1),
                       np.mean(temp_val_F2), model)
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        if early_stopping.early_stop:
            print("Early stopping")
            with open(log_file, 'a+', encoding='UTF-8') as fp:
                fp.write("Early stopping! Best Accs: {:.4f}\n\n\n".format(early_stopping.accs))
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2

lr = args.lr
weight_decay = args.weight_decay
patience = args.patience
n_epochs = args.epochs
batchsize = args.batch_size
datasetname = args.dataset
iterations = args.iter
modelname = args.modelname
hid_feats = args.hid_feats
out_feats = args.out_feats
image_size = args.image_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if args.mode == 'train':
        test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []
        print(device, args.gpu)
        tim = datetime.datetime.now()
        this_tim = str(tim.month) + '_' + str(tim.day) + '_' + str(tim.hour) + '_' + str(tim.minute) + '_' + str(
            tim.second)
        log_file = '../../log/' + datasetname + modelname + this_tim + '.txt'
        with open(log_file, 'a+', encoding='UTF-8') as fp:
            fp.write('VGA:lr:{}\tweight_decay:{}\tbatchsize:{}\tFusion:{}\n'.format(lr, weight_decay, batchsize,
                                                                                    args.fusion))

        for iter in range(iterations):
            torch.cuda.empty_cache()
            fold0_x_test, fold0_x_train, \
                fold1_x_test, fold1_x_train, \
                fold2_x_test, fold2_x_train, \
                fold3_x_test, fold3_x_train, \
                fold4_x_test, fold4_x_train = load5foldData()
            train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_VGA(
                fold0_x_test,
                fold0_x_train,
                lr, weight_decay,
                patience,
                n_epochs,
                batchsize,
                iter)
            train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_VGA(
                fold1_x_test,
                fold1_x_train, lr,
                weight_decay,
                patience,
                n_epochs,
                batchsize,
                iter)
            train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_VGA(
                fold2_x_test,
                fold2_x_train, lr,
                weight_decay,
                patience,
                n_epochs,
                batchsize,
                iter)
            train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_VGA(
                fold3_x_test,
                fold3_x_train, lr,
                weight_decay,
                patience,
                n_epochs,
                batchsize,
                iter)
            train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_VGA(
                fold4_x_test,
                fold4_x_train, lr,
                weight_decay,
                patience,
                n_epochs,
                batchsize,
                iter)
            test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
            ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
            ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
            PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
            PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
            REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
            REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
            F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
            F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        print("{}:|Total_Test_Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
              "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(args.dataset, sum(test_accs) / iterations,
                                                                        sum(ACC1) / iterations,
                                                                        sum(ACC2) / iterations, sum(PRE1) / iterations,
                                                                        sum(PRE2) / iterations,
                                                                        sum(REC1) / iterations, sum(REC2) / iterations,
                                                                        sum(F1) / iterations, sum(F2) / iterations))
        with open(log_file, 'a+', encoding='UTF-8') as fp:
            fp.write("{}:|Total_Test_Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                     "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}\n".format(args.dataset,
                                                                                 sum(test_accs) / iterations,
                                                                                 sum(ACC1) / iterations,
                                                                                 sum(ACC2) / iterations,
                                                                                 sum(PRE1) / iterations,
                                                                                 sum(PRE2) / iterations,
                                                                                 sum(REC1) / iterations,
                                                                                 sum(REC2) / iterations,
                                                                                 sum(F1) / iterations,
                                                                                 sum(F2) / iterations))
    elif args.mode == 'test':
        test_tim = datetime.datetime.now()
        test_log_file = '../../log/' + args.dataset + '_Test_' + str(test_tim.month) + '_' + str(test_tim.day) + '_' + \
                        str(test_tim.hour) + '_' + str(test_tim.minute) + '_' + str(test_tim.second) + '.txt'

        with open(test_log_file, 'a+', encoding='UTF-8') as fp:
            fp.write('Test Parameters:|modelname:{}|batchsize:{}|dataset:{}|\n'.format(
                args.modelname, args.batch_size, args.dataset))

        test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData()

        fold0_x_test += fold0_x_train
        fold1_x_test += fold1_x_train
        fold2_x_test += fold2_x_train
        fold3_x_test += fold3_x_train
        fold4_x_test += fold4_x_train

        test_folds = [fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test]  # 需要预先定义测试集

        for fold_idx, (model_path, x_test) in enumerate(zip(model_paths, test_folds)):
            print(f'Testing fold {fold_idx}')
            accs, acc1, pre1, rec1, f1, acc2, pre2, rec2, f2 = test_VGA(x_test, model_path)

            test_accs.append(accs)
            ACC1.append(acc1)
            ACC2.append(acc2)
            PRE1.append(pre1)
            PRE2.append(pre2)
            REC1.append(rec1)
            REC2.append(rec2)
            F1.append(f1)
            F2.append(f2)

        with open(test_log_file, 'a+', encoding='UTF-8') as fp:
            fp.write('\n\nFinal Average Results ====================\n')
            fp.write("Total_Test_Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}\n".format(
                np.mean(test_accs), np.mean(ACC1), np.mean(ACC2)))
            fp.write("pre1: {:.4f}|pre2: {:.4f}|rec1: {:.4f}|rec2: {:.4f}\n".format(
                np.mean(PRE1), np.mean(PRE2), np.mean(REC1), np.mean(REC2)))
            fp.write("F1: {:.4f}|F2: {:.4f}\n".format(np.mean(F1), np.mean(F2)))