import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import shutil
from model import *
from utils import *
from data import *
import argparse


def test_step(args, model, test_data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    for i, (dec_inputs, dec_outputs) in enumerate(test_data_loader):  
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs) 
        loss = criterion(outputs.view(args.batch_size, args.seq_len, args.vocab_size)[:,-1,:], dec_outputs[:,-1].view(-1))
        epoch_loss += loss.item()

    return epoch_loss / len(test_data_loader)


def last_word_acc(model, data, seq_len):
    # 如果预测的最后一个词跟句子的最后一个词一样，视为预测正确
    model.eval()
    correct = 0
    for sentence in data:
        output = model.test(sentence[:seq_len])
        if output == sentence[seq_len]:
            correct += 1.0

    return correct / len(data)

def get_test_acc(args):
    # 创建模型并加载参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_state_dict = torch.load('result/GPT_复合函数_学习率1e-4/composition-seed_1-N_10000/model/model_2999.pt')
    model_state_dict = torch.load('result/GPT_复合函数_4层_warmup_cos_lr_2e-5/composition-seed_1-N_500/model/model_200.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_7000/model/model_2999.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_4000/model/model_2999.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_10000/model/model_100.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_2000/model/model_2999.pt')
    model = myGPT(args, device)
    model.load_state_dict(model_state_dict)
    model.to(device)

    train_data_loader, test_data_loader, train_seq_group, test_seq_group, special_type_list_train, special_type_list_test = get_data(args)


    acc = last_word_acc(model, special_type_list_test, 9)

    print(100*acc, '%')


def find_first_occurrence_index(input_list):
    target_values = [1, 2, 3, 4]  # 你要查找的目标值
    for index, item in enumerate(input_list):
        if item in target_values:
            return index

def seq_rule(model, data, seq_len):
    # 如果预测的最后一个词跟句子的最后一个词一样，视为预测正确
    model.eval()
    correct = 0
    pair_all=[]
    for sentence in data:
        # print('input:', sentence)
        # output = model.test(sentence[:seq_len])
        output = model.test_diff(sentence[:seq_len])
        # print('output:', output)
        index=find_first_occurrence_index(sentence)
        delta=output[index:]-output[index-1:-1]
        delta=delta.tolist()
        delta=np.insert(delta,0, output[index-1].item()-sentence[index-1])
        last_non_zero_index = next((i for i, x in enumerate(reversed(delta)) if x), None)

        # 如果找到了非零元素，删除列表的最后一部分
        if last_non_zero_index is not None:
            delta = delta[:len(delta)-last_non_zero_index]
        pair_all.append(tuple(delta))

    counter = Counter(pair_all)

    # 按频次大小排序
    sorted_counts = counter.most_common()

    # 创建一个字典，key 是频次大小排序后的顺序，value 是元素和频次
    result = {i: (element, count) for i, (element, count) in enumerate(sorted_counts, 1)}
    print(str(sentence[index:index+2]), result)
    quit()
    return correct / len(data)


def each_layer_output(args):
    # 创建模型并加载参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_state_dict = torch.load('result/GPT_复合函数_学习率1e-4/composition-seed_1-N_10000/model/model_2999.pt')
    # model_state_dict = torch.load('result/GPT/3x_to_x-seed_1-N_800/model/model_3999.pt')
    model_state_dict = torch.load('result/GPT_复合函数_4层_warmup_cos_lr_2e-5/composition-seed_1-N_500/model/model_3999.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_7000/model/model_2999.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_4000/model/model_2999.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_10000/model/model_100.pt')
    # model_state_dict = torch.load('result/GPT_复合函数/composition-seed_1-N_2000/model/model_2999.pt')
    model = myGPT_new(args, device)
    model.load_state_dict(model_state_dict)
    model.to(device)

    args.test_data_size=100
    args.data_min = 20
    train_data_loader, test_data_loader, train_seq_group, test_seq_group, special_type_list_train, special_type_list_test = get_data(args)

    model.eval()
    for i, sentence in enumerate(special_type_list_test):
        output_list = model.test(sentence[:args.seq_len])
        if output_list[-1] == sentence[args.seq_len]:
            print(sentence, f'{i:<4} emb, emb+pos, 4层输出', '\033[0;33;40m', output_list, '\033[0m', '正确答案', sentence[args.seq_len], '--> 预测正确')
        else:
            print(sentence, f'{i:<4} emb, emb+pos, 4层输出', '\033[0;31;40m', output_list, '\033[0m', '正确答案', sentence[args.seq_len], '--> 预测错误')
        
            


    

if __name__ == '__main__':
    # args = read_json_data('result/GPT_复合函数_学习率1e-4/composition-seed_1-N_10000/config.json')
    # args = read_json_data('result/GPT/3x_to_x-seed_1-N_800/config.json')
    # args = read_json_data('result/GPT_复合函数/composition-seed_1-N_7000/config.json')
    # args = read_json_data('result/GPT_复合函数/composition-seed_1-N_4000/config.json')
    # args = read_json_data('result/GPT_复合函数/composition-seed_1-N_2000/config.json')
    args = read_json_data('result/GPT_复合函数_4层_warmup_cos_lr_2e-5/composition-seed_1-N_500/config.json')
    args = argparse.Namespace(**args)
    # args.target = 'context'
    # args.target = '3x_to_x'
    # args.target = 'composition'
    # train_data_loader, test_data_loader, train_seq_group, test_seq_group, special_type_list_train, special_type_list_test = get_data(args)

    # print(special_type_list_test)

    each_layer_output(args)
    # get_test_acc(args)













