from .DNN import myDNN
from .LSTM import myLSTM
from .GPT import myGPT
from .GPT_lightly import myGPT_lightly
from .GPT_specific import myGPT_specific
from .GPT_separate_attn_proj import myGPT_separate_attn_proj
from .GPT_separate_attn_proj_specific import myGPT_separate_attn_proj_specific
from .GPT_softmax10x import myGPT_softmax10x
from .GPT_OneHotEmb import myGPT_OneHotEmb

def get_model(args, device):
    if args.model == 'LSTM':
        model = myLSTM(args, device).to(device)
    elif args.model == 'GPT':
        model = myGPT(args, device).to(device)
    elif args.model == 'DNN':
        model = myDNN(args, device).to(device)
    elif args.model == 'GPT_lightly':
        model = myGPT_lightly(args, device).to(device)
    elif args.model == 'GPT_specific':
        model = myGPT_specific(args, device).to(device)
    elif args.model == 'GPT_separate_attn_proj':
        model = myGPT_separate_attn_proj(args, device).to(device)
    elif args.model == 'GPT_separate_attn_proj_specific':
        model = myGPT_separate_attn_proj_specific(args, device).to(device)
    elif args.model == 'GPT_softmax10x':
        model = myGPT_softmax10x(args, device).to(device)
    elif args.model == 'GPT_OneHotEmb':
        model = myGPT_OneHotEmb(args, device).to(device)
    return model