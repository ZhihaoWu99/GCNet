"""
@Project   : MVGPCA_v3
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : args.py
"""
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device, cuda:num or cpu")
    parser.add_argument("--path", type=str, default="./datasets/", help="Dataset path")
    parser.add_argument("--dataset", type=str, default="UAI", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default 42 (Vanilla GCN)")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="fix the seed or not")
    parser.add_argument("--n_repeated", type=int, default=5, help="Repeated times")

    parser.add_argument("--model", type=str, default='GCNet', choices=['GCN', 'GCNet'], help="Choose models")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--bias", action='store_true', default=False, help="Bias")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    parser.add_argument("--num_epoch", type=int, default=500, help="Training epochs")
    parser.add_argument("--num_pse", type=int, default=500, help="# pseudo labels")
    parser.add_argument("--alpha", type=float, default=5e-2, help="Hyperparameter alpha")
    parser.add_argument("--beta", type=float, default=3e-2, help="Hyperparameter beta")
    parser.add_argument("--p", type=float, default=1.5, help="Hyperparameter p")

    parser.add_argument("--layer", type=int, default=2, help="# layers")
    parser.add_argument("--hdim", type=int, default=16, help="Hidden dims")

    args = parser.parse_args()

    return args