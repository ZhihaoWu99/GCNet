import os
import warnings
import random
import numpy as np
import torch
from utils import tab_printer, set_args
from train import train


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = set_args()
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)

    all_ACC = []
    all_F1 = []
    all_TIME = []

    for i in range(args.n_repeated):
        ACC, F1, Time = train(args)
        all_ACC.append(ACC)
        all_F1.append(F1)
        all_TIME.append(Time)

    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC), np.std(all_ACC)))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1), np.std(all_F1)))