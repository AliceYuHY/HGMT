import torch, os, pickle, random
import numpy as np
from yaml import safe_load as yaml_load
from json import dumps as json_dumps
import scipy.sparse as sp


class DataLoad():
    def __init__(self, files):
        self.num_users = 0
        self.num_items = 0
        self.files = files

        files = self.files
        test_data = self.read_file(f'{files}/test.txt')
        train_data = self.read_file(f'{files}/train.txt')

        # 计算用户和物品的数量
        num_users = max(max(test_data.keys()), max(train_data.keys())) + 1
        num_items = max(max(item for items in test_data.values() for item in items),
                        max(item for items in train_data.values() for item in items)) + 1

        # 创建测试集和训练集的稀疏邻接矩阵
        self.test_adj_matrix = self.create_sparse_adjacency_matrix(test_data, num_users, num_items)
        self.train_adj_matrix = self.create_sparse_adjacency_matrix(train_data, num_users, num_items)
        self.num_users = num_users
        self.num_items = num_items


    # def __call__(self):
    #     return self.train_adj_matrix, self.test_adj_matrix

    def read_file(self, data_path):
        user_items = {}
        with open(data_path, 'r') as file:
            for line in file:
                line = line.strip().split()
                user = int(line[0])
                items = [int(x) for x in line[1:]]
                if user in user_items:
                    user_items[user].extend(items)
                else:
                    user_items[user] = items
        return user_items

    def create_sparse_adjacency_matrix(self, data, num_users, num_items):
        rows, cols = [], []
        for user, items in data.items():
            rows.extend([user] * len(items))
            cols.extend(items)
        values = [1] * len(rows)
        matrix = sp.csr_matrix((values, (rows, cols)), shape=(num_users, num_items), dtype=int)
        return matrix


def save_model(model, save_path, optimizer=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data2save = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(data2save, save_path)


def load_model(model, load_path, optimizer=None):
    data2load = torch.load(load_path, map_location='cpu')
    model.load_state_dict(data2load['state_dict'])
    if optimizer is not None and data2load['optimizer'] is not None:
        optimizer = data2load['optimizer']


def fix_random_seed_as(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    pass
