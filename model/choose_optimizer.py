import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR



def get_optimizer(model, args, **kwargs):
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    scheduler = None

    return optimizer, scheduler