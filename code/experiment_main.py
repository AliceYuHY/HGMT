import pandas as pd
import torch, pickle, time, os
import numpy as np
from options import parse_args
from torch.utils.data import DataLoader
from model import GHMT, BPRLoss
from preprocess import prepare_graph_data, UserItemDataset
from utils import DataLoad, load_model, save_model, fix_random_seed_as
from tqdm import tqdm
import joblib
from Trainmodel import Train_Model

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":

    if __name__ == "__main__":
        # Parse command line arguments
        args = parse_args()

        # Define the list of datasets to process
        datasets = ['ifashion', 'yelp2018', 'amazon-book']

        # Number of repetitions for the experiments
        num_repeats = 10

        # Loop over each repetition
        for repeat_index in range(num_repeats):
            # Process each dataset
            for dataset_name in datasets:
                # Set up the path to the dataset
                args.data_path = f'../data/{dataset_name}/'
                args.data_name = dataset_name

                # Set the negative sampling size and number of epochs
                args.neg_size = 200
                args.n_epoch = 200

                # Load the dataset
                data_loader = DataLoad(args.data_path)

                # Create training and validation sets
                train_dataset = UserItemDataset(data_loader, 'train', args)
                val_dataset = UserItemDataset(data_loader, 'val', args)

                # Seed setting for reproducibility
                fix_random_seed_as(args.seed)

                # Initialize the model
                app = Train_Model(args, data_loader, train_dataset, val_dataset)

                # Train the model
                app.train()

                # Ensure the directory for saving results exists
                save_dir = './save_file'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                from Trainmodel import save_loss, save_top
                # Save the loss and performance metrics
                joblib.dump(save_loss, f'{save_dir}/{dataset_name}_save_loss.list')
                joblib.dump(save_top, f'{save_dir}/{dataset_name}_save_top.list')