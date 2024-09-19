import torch
import dgl
import numpy as np
import torch.utils.data


class UserItemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, set_type='train', configuration=None):
        super(UserItemDataset, self).__init__()
        if set_type not in ['train', 'val']:
            raise ValueError(f'Invalid set_type {set_type}')
        self.configuration = configuration
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.is_training = set_type == 'train'

        if self.is_training:
            user_ids, item_ids = dataset.train_adj_matrix.nonzero()
            self.pairs = np.stack((user_ids, item_ids), axis=1).astype(np.int64)
            self.active_pairs_set = set((uid, iid) for uid, iid in zip(user_ids, item_ids))
        else:
            user_ids, item_ids = dataset.test_adj_matrix.nonzero()
            self.active_pairs_set = set((uid, iid) for uid, iid in zip(user_ids, item_ids))
            validation_data = []
            for user_id, positive_item in zip(user_ids, item_ids):
                validation_data.append((user_id, positive_item))
                random_negatives = np.random.randint(0, self.num_items, size=self.configuration.neg_size,
                                                     dtype=np.int64)
                for negative_item in random_negatives:
                    while (user_id, negative_item) in self.active_pairs_set:
                        negative_item = np.random.randint(0, self.num_items)
                    validation_data.append((user_id, negative_item))
            self.pairs = np.array(validation_data, dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        user, item = self.pairs[index]
        if self.is_training:
            negative_item = self.neg_data[index]
            return user, item, negative_item
        else:
            return user, item


def prepare_graph_data(configuration, dataset):
    source_nodes = []
    destination_nodes = []
    edge_types = []

    user_offset = 0
    item_offset = user_offset + dataset.num_users
    node_count = item_offset + dataset.num_items

    # User-Item interactions
    user_ids, item_ids = dataset.train_adj_matrix.nonzero()
    source_nodes += (item_offset + item_ids).tolist()
    destination_nodes += (user_offset + user_ids).tolist()
    edge_types += [1] * dataset.train_adj_matrix.nnz

    # Reverse for symmetry
    source_nodes += (user_offset + user_ids).tolist()
    destination_nodes += (item_offset + item_ids).tolist()
    edge_types += [2] * dataset.train_adj_matrix.nnz

    graph = dgl.graph((source_nodes, destination_nodes), num_nodes=node_count)
    graph.edata['type'] = torch.LongTensor(edge_types)

    # Adding self-loops to all nodes
    self_loops = torch.arange(node_count)
    graph.add_edges(self_loops, self_loops)

    return graph
