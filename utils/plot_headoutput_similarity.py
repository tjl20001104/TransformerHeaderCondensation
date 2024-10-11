import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from collections import defaultdict
from .plot_settings import *
import seaborn as sns
from torch.nn import functional as F
import math
from utils import *
import argparse

def plot_similarity_with_input(working_dir):
    r'''绘制一个实验中具体类型数据的acc随epoch变化的曲线'''
    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(wspace=0.4, hspace=0.6, bottom=0.16, fs=12, lw=6, ms=12.5, axlw=2.5, major_tick_len=10)
    
    model_output_similarity_his = np.load(f'{working_dir}/similarity/similarity_his.npy', allow_pickle=True).item()
    for type_s, v1 in model_output_similarity_his.items():
        sample = v1['sample']
        similaritys = v1['similarity']
        for epoch, v2 in similaritys.items():
            for layer_id, v3 in v2.items():
                file_name = '{}/pic/heatmap_attn_output_similarity/{}/layer_{}/epoch_{}.png'
                dir_name = '{}/pic/heatmap_attn_output_similarity/{}/layer_{}/'.format(working_dir, type_s, layer_id)
                os.makedirs(dir_name, exist_ok=True)
                plt.title('similarity heat map of layer {} at epoch {}\n{} sample:{}'.format(layer_id, epoch, type_s, sample))
                plt.imshow(v3)
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(file_name.format(working_dir, type_s, layer_id, epoch))
                plt.close()



def compute_similarity(X_input, model, device, args):
    with torch.no_grad():
        X_input = X_input.to(device)
        decoder_layers = model.decoder.layers
        hidden_state = model.embedding(X_input)
        dec_self_attn_mask = attn_mask(X_input, device)
        res_decoder = dict()

        for i in range(len(decoder_layers)):
            attn = decoder_layers[i].dec_self_attn
            qxs = attn.W_Q(hidden_state).chunk(args.n_heads, dim=-1) 
            kxs = attn.W_K(hidden_state).chunk(args.n_heads, dim=-1) 
            vxs = attn.W_V(hidden_state).chunk(args.n_heads, dim=-1)
            hidden_state,_ = decoder_layers[i](hidden_state, dec_self_attn_mask)

            res_decoder[i] = cos_similarity_attention(dec_self_attn_mask, qxs, kxs, vxs, args)

    return res_decoder

def cos_similarity_attention(attn_mask, qxs, kxs, vxs, args):
    outputs = []
    if len(qxs) <= 1:
        return None
    length = len(qxs)
    for i in range(length):
        outputs.append(attention(attn_mask, qxs[i], kxs[i], vxs[i])[0])

    heat_map = np.ones((length,length))
    for i in range(length):
        for j in range(i+1, length):
            mat1 = outputs[i].reshape(-1,1).squeeze()
            mat2 = outputs[j].reshape(-1,1).squeeze()
            # cos = torch.cosine_similarity(mat1.unsqueeze(1),mat2.unsqueeze(0),dim=-1)
            cos = torch.cosine_similarity(mat1, mat2, dim=0)
            heat_map[i][j] = cos.detach().cpu().numpy()
            heat_map[j][i] = cos.detach().cpu().numpy()

    heat_map = heat_map_order_exchange(heat_map, args)
    return heat_map

def heat_map_order_exchange(heat_map, args):
    order = []
    size = heat_map.shape[0]
    if size <= 8:
        return heat_map
    order1 = list(range(size))
    k = 0
    order_temp = []
    threshold = args.plot_headoutput_similarity_threshold

    order1 = []
    for j in range(size):
        if j != 0:
            for i in order2:
                if heat_map[k][i] > threshold:
                    order.append(i)
                else:
                    order1.append(i)
        else:
            for i in range(size):
                if heat_map[0][i] > threshold:
                    order.append(i)
                else:
                    order1.append(i)

        order_temp = order_temp + order
        if len(order_temp) == size:
            break
        k = order1[0]
        order2 = order1
        order1 = []
        order = []

    # for j in range(int(size/4)):
    #     order = list(np.argsort(-heat_map[k,order1])[:4])
    #     order_add = np.array(order1)[order]
    #     order_temp = order_temp + list(order_add)
    #     for o in order_add:
    #         order1.remove(o)
    #     if len(order1) > 0:
    #         k = order1[0]
    
    heat_map = heat_map[order_temp,:]
    heat_map = heat_map[:,order_temp]

    return heat_map

def attention(attn_mask, query, key, value):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    masked_attn = scores.masked_fill(attn_mask, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(masked_attn, dim=-1)

    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


def get_attn_pad_mask(seq_q, seq_k):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq, device):
    '''
        seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

def attn_mask(X_input, device):
    '''
        X_input: [batch_size, tgt_len]
    '''
    dec_self_attn_pad_mask = get_attn_pad_mask(X_input, X_input) # [batch_size, tgt_len, d_model] 遮挡padding部分
    dec_self_attn_subsequence_mask = get_attn_subsequence_mask(X_input, device) # [batch_size, tgt_len, d_model] 遮挡未来时刻的词
    # 两个mask之和只要有一个为1的地方，就为1
    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, d_model] 

    return dec_self_attn_mask