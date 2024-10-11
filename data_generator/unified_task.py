import torch
import torch.utils.data as Data
import numpy as np
import random

def tasks_multi_composition(args, seq, dataset, mode='train'):
    '''
        prompt:
            在句子前半部分生成prompt 9，9之后的数视为x，x之后随机位置生成若干个运算类prompt
            运算类prompt:
                prompt 1: 1x to x + 1
                prompt 2: 2x to x + 3
                prompt 3: 3x to x - 2       
                prompt 4: 4x to x - 3     
    '''
    def simple_operate(x, prompt):
        if prompt == 1:
            return x + 1
        elif prompt == 2:
            return x + 3
        elif prompt == 3:
            return x - 2
        elif prompt == 4:
            return x - 3
    
    calculate_prompt_list = [1,2,3,4]  # 可选的prompt

    
    pos_1 = np.random.randint(0, args.seq_len//2-1)
    # # 在pos_1的下一位x之后，随机生成若干个运算类prompt
    # mode形式为 'ab推cd_train' 或 'ab推cd_test'形式，若后缀为train则生成a或b个prompt，若后缀为test则生成c个或d个prompt
    prefix, suffix = mode.split('_')
    anchor_num1, anchor_num2 = prefix.split('推')
    # 将anchor_num1切分为单个字符，并转为int类型
    anchor_num1 = list(anchor_num1)
    anchor_num1 = list(map(int, anchor_num1))
    # 将anchor_num2切分为单个字符，并转为int类型
    anchor_num2 = list(anchor_num2)
    anchor_num2 = list(map(int, anchor_num2))

    if suffix == 'train':
        prompt_num_I1 = random.choice(anchor_num1)
    elif suffix == 'test':
        prompt_num_I1 = random.choice(anchor_num2)
    
        
    seq[pos_1] = 9
    x = random.choice(dataset[str((pos_1+1) % 8)])
    seq[pos_1+1] = x

    prompt_list_I1 = random.choices(calculate_prompt_list, k=prompt_num_I1)
    pos_list = []
    for prompt in prompt_list_I1:
        while True:
            pos = np.random.randint(pos_1+2, args.seq_len)
            if pos not in pos_list:
                break
        pos_list.append(pos)
        seq[pos] = prompt

    for prompt in prompt_list_I1:
        x = simple_operate(x, prompt)
    seq[-1] = x

    return seq


def tasks_unified(args, seq, dataset, mode='.'):
    '''
        prompt:
            识别key item类prompt:
                prompt 201: 201之后的x视为x1
                prompt 202: 202之后的x视为x2
                prompt 203: prompt 3之前的prompt作用于x1, 之后的prompt作用于x2
            运算类prompt:
                prompt 204: 204x to x + 1
                prompt 205: 205x to x + 3
                prompt 206: 206x to x - 2       
                prompt 207: 207x to x - 3     
            输出类prompt(一个seq中仅出现1个):
                prompt 208: 末尾是208则输出x1运算后的结果f_1(x1)
                prompt 209: 末尾是209则输出x2运算后的结果f_2(x2)
                prompt 210: 末尾是210则输出x1和x2运算后结果的均值取整 [f_1(x1) + f_2(x2) / 2]
    '''
    def simple_operate(x, prompt):
        if prompt == 204:
            return x + 1
        elif prompt == 205:
            return x + 3
        elif prompt == 206:
            return x - 2
        elif prompt == 207:
            return x - 3
    
    calculate_prompt_list = [204, 205, 206, 207]  # 可选的prompt

    while True:
        # 在中间某个随机位置生成prompt 203
        pos_3 = np.random.randint(4, args.seq_len-5)
        # 在prompt 203之前的位置生成prompt 201，与prompt 203至少间隔1个位置
        pos_1 = np.random.randint(0, pos_3-1)
        # 在0到pos_203之间，但不是pos_201和pos_x1位置，随机生成若干个运算类prompt
        prompt_num_I1 = np.random.randint(1, pos_3-1)  # 生成的prompt数量
        prompt_num_I1 = min(prompt_num_I1, 4)
        # 在prompt 203之后的位置生成prompt 202，与最后一位至少间隔3个位置
        pos_2 = np.random.randint(pos_3+1, args.seq_len-3)
        # 在pos_203+1到args.seq_len-3之间，但不是pos_2和pos_x2位置，随机生成若干个运算类prompt
        prompt_num_I2 = np.random.randint(1, args.seq_len-3-pos_3)  # 生成的prompt数量
        prompt_num_I2 = min(prompt_num_I2, 4)

        if mode == 'train_task':
            if prompt_num_I1 != 2 and prompt_num_I2 != 2:
                break
        elif mode == 'test_task':
            if prompt_num_I1 == 2 and prompt_num_I2 == 2:
                break


    seq[pos_3] = 203

    # 对prompt 203前边进行操作
    pos_x1 = pos_1 + 1
    x1 = random.choice(dataset[str((pos_x1) % 8)])
    seq[pos_1] = 201
    seq[pos_x1] = x1

    # 在0到pos_203之间，但不是pos_201和pos_x1位置，随机生成若干个运算类prompt
    prompt_list_I1 = random.choices(calculate_prompt_list, k=prompt_num_I1)
    pos_list = []
    for prompt in prompt_list_I1:
        while True:
            pos = np.random.randint(0, pos_3)
            if pos != pos_1 and pos != pos_x1 and pos not in pos_list:
                break
        pos_list.append(pos)
        seq[pos] = prompt

    # 对prompt 203后边进行操作
    pos_x2 = pos_2 + 1
    x2 = random.choice(dataset[str((pos_x2) % 8)])
    seq[pos_2] = 202
    seq[pos_x2] = x2

    prompt_list_I2 = random.choices(calculate_prompt_list, k=prompt_num_I2)
    pos_list = []
    for prompt in prompt_list_I2:
        while True:
            pos = np.random.randint(pos_3+1, args.seq_len-1)
            if pos != pos_2 and pos != pos_x2 and pos not in pos_list:
                break
        pos_list.append(pos)
        seq[pos] = prompt

    # 倒数第二位生成prompt 207-213
    pos_out_prompt = args.seq_len - 1
    out_prompt = np.random.randint(207, 214)
    seq[pos_out_prompt] = out_prompt

    # 倒数第一位生成运算后的结果
    if out_prompt == 208:
        for prompt in prompt_list_I1:
            x1 = simple_operate(x1, prompt)
        seq[-1] = x1
    
    elif out_prompt == 209:
        for prompt in prompt_list_I2:
            x2 = simple_operate(x2, prompt)
        seq[-1] = x2
    
    elif out_prompt == 210:
        for prompt in prompt_list_I1:
            x1 = simple_operate(x1, prompt)
        for prompt in prompt_list_I2:
            x2 = simple_operate(x2, prompt)
        seq[-1] = int((x1 + x2) / 2)

    return seq









