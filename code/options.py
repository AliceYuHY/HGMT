import os, argparse
from yaml import safe_load as yaml_load
from json import dumps as json_dumps

def parse_args(show_args=True):
    parser = argparse.ArgumentParser(description='GHMT Model Arguments')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device number')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for reproducibility')
    parser.add_argument('--n_hid', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--mem_size', type=int, default=8, help='Size of memory component')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in the transformer')
    parser.add_argument('--feat_drop', type=float, default=0.6, help='Feature dropout rate')
    parser.add_argument('--attn_drop', type=float, default=0.6, help='Attention dropout rate')
    parser.add_argument('--residual', action='store_true', help='Whether to include residual connections')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--reg', type=float, default=0.0005, help='Regularization strength')
    parser.add_argument('--decay', type=float, default=0.985, help='Decay rate for learning rate scheduler')
    parser.add_argument('--decay_step', type=int, default=1, help='Step size for learning rate decay')
    parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10240, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--dataset', type=str, default="", help='Dataset name')
    parser.add_argument('--data_path', type=str, default="", help='Path to the dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--checkpoint', type=str, default="", help='Path to the checkpoint file')
    parser.add_argument('--model_dir', type=str, default="", help='Directory to save model files')
    parser.add_argument('--repeat', type=int, default=1, help='Directory to save model files')

    args = parser.parse_args()

    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if show_args:
        print('Arguments:\n{}'.format(json_dumps(args.__dict__, indent=4)), flush=True)

    return args
