import torch
import torch.utils.data as Data
from torch import nn
import numpy as np


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


class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask, d_k):
            '''
                Q: [batch_size, n_heads, len_q, d_k]
                K: [batch_size, n_heads, len_k, d_k]
                V: [batch_size, n_heads, len_v(=len_k), d_v]
                attn_mask: [batch_size, n_heads, seq_len, seq_len]
            '''
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
                d_k)  # scores : [batch_size, n_heads, len_q, len_k]
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
            return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.n_head = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model
    
        self.W_Q1 = nn.Linear(args.d_model//2, args.d_k * args.n_heads//2, bias=False)
        self.W_K1 = nn.Linear(args.d_model//2, args.d_k * args.n_heads//2, bias=False)
        self.W_V1 = nn.Linear(args.d_model//2, args.d_v * args.n_heads//2, bias=False)
        self.fc1 = nn.Linear(args.n_heads * args.d_v//2, args.d_model//2, bias=False)

        self.W_Q2 = nn.Linear(args.d_model//2, args.d_k * args.n_heads//2, bias=False)
        self.W_K2 = nn.Linear(args.d_model//2, args.d_k * args.n_heads//2, bias=False)
        self.W_V2 = nn.Linear(args.d_model//2, args.d_v * args.n_heads//2, bias=False)
        self.fc2 = nn.Linear(args.n_heads * args.d_v//2, args.d_model//2, bias=False)
        self.layernorm = nn.LayerNorm(args.d_model//2)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # residual, batch_size = input_Q, input_Q.size(0)
        batch_size = input_Q.size(0)
        res1 = input_Q[:, :, :self.d_model//2]
        res2 = input_Q[:, :, self.d_model//2:]
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q1 = self.W_Q1(input_Q[:, :, :self.d_model//2]).view(batch_size, -1, self.n_head, self.d_k//2).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K1 = self.W_K1(input_K[:, :, :self.d_model//2]).view(batch_size, -1, self.n_head, self.d_k//2).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V1 = self.W_V1(input_V[:, :, :self.d_model//2]).view(batch_size, -1, self.n_head, self.d_v//2).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        Q2 = self.W_Q2(input_Q[:, :, self.d_model//2:]).view(batch_size, -1, self.n_head, self.d_k//2).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K2 = self.W_K2(input_K[:, :, self.d_model//2:]).view(batch_size, -1, self.n_head, self.d_k//2).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V2 = self.W_V2(input_V[:, :, self.d_model//2:]).view(batch_size, -1, self.n_head, self.d_v//2).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        Q = torch.cat([Q1, Q2], dim=-1)
        K = torch.cat([K1, K2], dim=-1)
        V = torch.cat([V1, V2], dim=-1)


        # Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        # K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        # V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # scores : [batch_size, n_heads, len_q, len_k]
        attn = torch.matmul(Q1, K1.transpose(-1, -2)) / np.sqrt(self.d_k//2)
        masked_attn = attn.masked_fill(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        softmax_attn = nn.Softmax(dim=-1)(masked_attn)
        qkv = torch.matmul(softmax_attn, V)  # [batch_size, n_heads, len_q, d_v] 

        qkv_out = qkv.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        # print(qkv_out[:, :, :self.d_model//2].shape)
        # output = self.fc(qkv_out)  # [batch_size, len_q, d_model]
        output1 = self.layernorm(self.fc1(qkv_out[:, :, :self.n_head * self.d_v//2]) + res1)
        output2 = self.layernorm(self.fc2(qkv_out[:, :, self.n_head * self.d_v//2:]) + res2)
        output = torch.cat([output1, output2], dim=-1)

        return output, softmax_attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(args.d_model//2, args.d_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_feedforward, args.d_model//2, bias=False)
        )

        d = args.d_model - args.d_model//2
        self.fc2 = nn.Sequential(
            nn.Linear(d, args.d_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_feedforward, d, bias=False)
        )

        self.layernorm = nn.LayerNorm(args.d_model//2)

        self.d_model = args.d_model

    def forward(self, hidden_state):
        '''
            hidden_state: [batch_size, seq_len, d_model]
        '''
        residual = hidden_state
        # output = self.fc1(hidden_state)
        # d_model的前一半过fc1，后一半过fc2
        res1 = hidden_state[:, :, :self.d_model//2]
        res2 = hidden_state[:, :, self.d_model//2:]
        output1 = self.layernorm(self.fc1(hidden_state[:, :, :self.d_model//2]) + res1)
        output2 = self.layernorm(self.fc2(hidden_state[:, :, self.d_model//2:]) + res2)
        output = torch.cat([output1, output2], dim=-1)
        return output # [batch_size, seq_len, d_model]

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            hidden_state: [batch_size, tgt_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # Attention层
        # hidden_state: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        hidden_state, dec_self_attn = self.dec_self_attn(hidden_state, hidden_state, hidden_state, dec_self_attn_mask)

        # 非线性层
        hidden_state = self.pos_ffn(hidden_state)  # [batch_size, tgt_len, d_model]
        return hidden_state, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            hidden_state: [batch_size, tgt_len]
        '''
        dec_self_attns = []
        for layer in self.layers:
            # hidden_state: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            hidden_state, dec_self_attn = layer(hidden_state, dec_self_attn_mask)
   
            dec_self_attns.append(dec_self_attn)

        return hidden_state, dec_self_attns

class Embedding(nn.Module):
    def __init__ (self, args, device):
        super(Embedding, self).__init__()
        self.device = device
        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.max_pos, args.d_model)

    def forward(self, X_input):
        seq_len = X_input.size(1)
        pos = torch.arange(seq_len, dtype = torch.long, device = self.device)
        pos = pos.unsqueeze(0).expand_as(X_input)

        tgt_emb = self.tgt_emb(X_input)
        pos_emb = self.pos_emb(pos)
        emb = tgt_emb + pos_emb

        return emb

class myGPT_separate_attn_proj(nn.Module):
    def __init__(self, args, device):
        super(myGPT_separate_attn_proj, self).__init__()

        self.device = device
        self.embedding = Embedding(args, device)
        self.decoder = Decoder(args, device)
        self.projection = nn.Linear(args.d_model//2, args.vocab_size)

        self.d_model = args.d_model

    def forward(self, X_input):
        """
            dec_inputs: [batch_size, tgt_len]
        """
        hidden_state = self.embedding(X_input)

        dec_self_attn_mask = attn_mask(X_input, self.device)

        hidden_state, dec_self_attns = self.decoder(hidden_state, dec_self_attn_mask)
        
        dec_logits = self.projection(hidden_state[:,:,self.d_model//2:])
        
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns
    

    def greedy_decoder(self,dec_input):

        projected, _ = self.forward(dec_input)

        projected = projected[-1,:].argmax()
        next_word = projected.item() 

        return next_word


    def test(self,sentence):
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)

        output = self.greedy_decoder(dec_input)

        return output




