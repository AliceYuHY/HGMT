import pandas as pd
import torch, pickle, time, os
import numpy as np
from options import parse_args
from torch.utils.data import DataLoader
from model import GHMT, BPRLoss
from utils import DataLoad, load_model, save_model, fix_random_seed_as
from tqdm import tqdm
import joblib
from preprocess import prepare_graph_data, UserItemDataset

save_loss = []
save_top = []

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Train_Model():
    def __init__(self, config, data_loader, training_set, validation_set):
        """
        Initializes the training model with the necessary configurations and data.

        Args:
            config (object): Configuration object containing parameters like CUDA availability.
            data_loader (DataLoader): The data loader for managing data input during training.
            training_set (Dataset): Dataset object containing the training data.
            validation_set (Dataset): Dataset object containing the validation data.

        Attributes:
            device (torch.device): Device on which the model will run (GPU or CPU).
        """
        self.config = config
        self.device = torch.device('cuda' if config.cuda and torch.cuda.is_available() else 'cpu')

        # Initialize data loaders for training and validation datasets
        self.train_loader = DataLoader(
            dataset=training_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        self.validation_loader = DataLoader(
            dataset=validation_set,
            batch_size=config.test_batch_size * (config.neg_size + 1),
            shuffle=False,
            num_workers=config.num_workers
        )

        # Prepare the graph structure for the model
        self.graph = prepare_dgl_graph(config, data_loader).to(self.device)

        # Initialize the model
        self.model = GHMT(config, data_loader.num_users, data_loader.num_items)
        self.model.to(self.device)

        # Setup the loss criterion and optimizer
        self.criterion = BPRLoss(config.reg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.decay_step,
            gamma=config.decay
        )

        # Load model from checkpoint if specified
        if config.checkpoint:
            load_model(self.model, config.checkpoint, self.optimizer)

        # Prepare user and item indices for generating recommendations
        self.sampled_user_indices = [torch.randint(0, self.model.n_user, (1,)).item() for _ in range(4)]
        self.user_tensor_list = []
        self.item_tensor_list = []
        for user_index in self.sampled_user_indices:
            self.user_tensor_list.append((user_index + torch.arange(self.model.n_item)).to(self.device))
            self.item_tensor_list.append((self.model.n_user + torch.arange(self.model.n_item)).to(self.device))

        self.dataset_name = config.data_name

    def train(self):
        config = self.args
        best_hr, best_ndcg, best_epoch, patience_counter = 0, 0, 0, 0
        training_start_time = time.time()

        for self.current_epoch in range(1, config.n_epoch + 1):
            epoch_losses = self.train_one_epoch(self.train_loader, self.graph)
            model_representation = self.model(self.graph)
            user_item_scores = {}

            # Score computation and saving
            for index, user_index in enumerate(self.sampled_user_indices):
                user_representation = model_representation[self.user_tensor_list[index]]
                item_representation = model_representation[self.item_tensor_list[index]]
                scores = self.model.predict(user_representation, item_representation)
                user_item_scores[user_index] = scores.cpu().detach().numpy()

            pd.DataFrame(user_item_scores).to_csv(f'{self.dataset_name}_epoch_{self.current_epoch}_scores.csv')
            torch.cuda.empty_cache()  # Free GPU memory

            save_loss.append([config.seed, self.current_epoch, epoch_losses])
            print(
                f'Epoch {self.current_epoch} completed! Time elapsed: {time.time() - training_start_time:.2f}s, Losses: {epoch_losses}',
                flush=True)

            # Validation at different top-k values
            for top_k in [5, 10, 15, 20, 50]:
                hr, ndcg, _ = self.validate(self.validation_loader, self.graph, top=top_k)
                save_top.append([config.seed, self.current_epoch, top_k, hr, ndcg])

                if hr + ndcg > best_hr + best_ndcg:
                    best_hr, best_ndcg, best_epoch = hr, ndcg, self.current_epoch
                    patience_counter = 0
                    if config.model_dir:
                        self.save_best_model(hr, ndcg)
                else:
                    patience_counter += 1

                if patience_counter >= config.patience:
                    print(f'Early stop at epoch {self.current_epoch}, best epoch {best_epoch}')
                    break

        print(f'Best N@{config.topk}: {best_ndcg:.4f}, Best R@{config.topk}: {best_hr:.4f}', flush=True)

    # Method to train the model for one epoch
    def train_one_epoch(self, dataloader, graph):
        self.model.train()
        total_loss = 0
        dataloader.dataset.neg_sample()  # Negative sampling for batch
        for batch in tqdm(dataloader, desc='Training'):
            user_idx, pos_idx, neg_idx = batch
            representations = self.model(graph)
            pos_preds = self.model.predict(representations[user_idx], representations[self.model.n_user + pos_idx])
            neg_preds = self.model.predict(representations[user_idx], representations[self.model.n_user + neg_idx])
            loss = self.criterion(pos_preds, neg_preds)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler:
            self.scheduler.step()

        return total_loss

    # Validation method for computing HR and NDCG
    def validate(self, dataloader, graph, top=5):
        self.model.eval()
        total_hr, total_ndcg = 0, 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validating'):
                user_idx, item_idx = batch
                representations = self.model(graph)
                preds = self.model.predict(representations[user_idx], representations[self.model.n_user + item_idx])
                hr, ndcg = self.calc_hr_and_ndcg(preds, top)
                total_hr += hr
                total_ndcg += ndcg
        return np.mean(total_hr), np.mean(total_ndcg)

    # Calculate Hit Rate (HR) and Normalized Discounted Cumulative Gain (NDCG)
    def calc_hr_and_ndcg(self, scores, topk):
        reshaped_scores = scores.reshape(-1, self.args.neg_size + 1)
        actual = torch.zeros_like(reshaped_scores)
        actual[:, 0] = 1  # First item is the positive one
        top_indices = scores.topk(topk, dim=1).indices
        hits = actual.gather(1, top_indices)
        hr = hits.sum(1).mean().item()
        ndcg = (hits * torch.log2(torch.arange(2, topk + 2)).inverse()).sum(1).mean().item()
        return hr, ndcg

    # Saving the best model configuration
    # Saving the best model configuration
    def save_best_model(self, hr, ndcg):
        description = f'{self.args.dataset}_hid_{self.args.n_hid}_layers_{self.args.n_layers}_mem_{self.args.mem_size}_' + \
                      f'lr_{self.args.lr}_reg_{self.args.reg}_decay_{self.args.decay}_step_{self.args.decay_step}_batch_{self.args.batch_size}'
        performance = f'HR_{hr:.4f}_NDCG_{ndcg:.4f}'
        filename = f'{description}_{performance}.pth'
        save_model(self.model, os.path.join(self.args.model_dir, filename), self.optimizer)
        print(f"Model saved with HR: {hr:.4f} and NDCG: {ndcg:.4f}")

    # Additional utility methods can be added here to handle things like model loading, additional metrics computation, or other training utilities.

    # Load a previously saved model checkpoint
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Checkpoint loaded successfully.")
        else:
            print("No checkpoint found at the specified path.")



