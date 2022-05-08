import os
import torch
from datetime import datetime
from experiments.exp_sci import Exp_SCINet
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='SCINet')
### -------  dataset settings --------------
parser.add_argument('--group', type=str, default='p',
                    choices=['p','pf','pr','pi'])
parser.add_argument('--data', type=str, default='../datasets/p5.csv',
                    help='location of the data file')
parser.add_argument('--normalize', type=int, default=3)  # 3是标准化，2是归一化

### -------  device settings --------------
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--window_size', type=int, default=168, help='input length')
parser.add_argument('--horizon', type=int, default=3, help='prediction length')
parser.add_argument('--concat_len', type=int, default=165)
parser.add_argument('--single_step', type=int, default=0, help='only supervise the final setp')
parser.add_argument('--single_step_output_One', type=int, default=0, help='only output the single final step')
parser.add_argument('--lastWeight', type=float, default=1.0, help='Loss weight lambda on the final step')
parser.add_argument('--tra_date', type=str, default="2018-01-02", help='the start date of training data of stock')
parser.add_argument('--val_date', type=str, default="2020-06-01", help='the start date of validation data of stock')
parser.add_argument('--tes_date', type=str, default="2020-12-31", help='the start date of test data of stock')


### -------  training settings --------------  
parser.add_argument('--train', action= 'store_true', default=False)
parser.add_argument('--resume', action= 'store_false', default=True)
parser.add_argument('--evaluate', action= 'store_false', default=True)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
# parser.add_argument('--save', type=str, default='model/pf5.pt',
#                     help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--num_nodes', type=int, default=8, help='number of nodes/variables')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
parser.add_argument('--save_path', type=str, default='../model_saved/SCINet/')
parser.add_argument('--model_name', type=str, default='SCINet')
parser.add_argument('--output_path', type=str, default='../res/SCINet')

### -------  model settings --------------  
parser.add_argument('--hidden-size', default=1.0, type=float, help='hidden channel of module')  # H, EXPANSION RATE
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size')  # k kernel size
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type=bool, default=False)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--stacks', type=int, default=1)
parser.add_argument('--fix', type=int, default=1)

args = parser.parse_args()

args.concat_len = args.window_size - args.horizon

if __name__ == '__main__':
    if args.fix ==1:
        torch.manual_seed(4321)  # reproducible
        torch.cuda.manual_seed_all(4321)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
        torch.backends.cudnn.enabled = True

    Exp = Exp_SCINet
    exp = Exp(args)
    print(args.train)

    if args.evaluate:
        data = exp._get_data()
        before_evaluation = datetime.now().timestamp()
        if args.stacks == 1:
            rse, rae, correlation = exp.validate(data, data.test[0], data.test[1], evaluate=True)
        else:
            rse, rae, correlation, rse_mid, rae_mid, correlation_mid = exp.validate(data, data.test[0], data.test[1],
                                                                                    evaluate=True)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

    elif args.train or args.resume:
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        normalize_statistic = exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")
        data = exp._get_data()
        exp.validate(data, data.test[0], data.test[1], evaluate=True)
