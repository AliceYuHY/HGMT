import os
import pandas as pd
import torch
import numpy as np
from options import parse_args
from torch.utils.data import DataLoader
from model import GHMT, BPRLoss
from preprocess import prepare_graph_data, UserItemDataset
from utils import DataLoad, load_model, save_model, fix_random_seed_as
from tqdm import tqdm
import joblib
from Trainmodel import Train_Model

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    # Parse command line arguments
    args = parse_args()

    # Setup the dataset and model configuration based on input arguments
    args.data_path = f'../data/{args.data_name}/'

    # Load the dataset
    data_loader = DataLoad(args.data_path)

    # Create training and validation sets
    train_dataset = UserItemDataset(data_loader, 'train', args)
    val_dataset = UserItemDataset(data_loader, 'val', args)

    # Seed setting for reproducibility
    fix_random_seed_as(args.seed)
    for r in range(args.repeats):
        # Initialize and train the model
        model_app = Train_Model(args, data_loader, train_dataset, val_dataset)
        model_app.train()

        # Ensure the directory for saving results exists
        save_dir = './save_file'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        from Trainmodel import save_loss, save_top

        # Save the loss and performance metrics
        joblib.dump(save_loss, f'{save_dir}/{args.data_name}_save_loss_{r}.list')
        joblib.dump(save_top, f'{save_dir}/{args.data_name}_save_top_{r}.list')

if __name__ == "__main__":
    main()
