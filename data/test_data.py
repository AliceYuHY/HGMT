import numpy as np
from scipy.sparse import coo_matrix


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    user_data = []
    for line in lines:
        line = line.strip().split()
        data.append([int(x) for x in line[1:]])
        user_data.append(int(line[0]))
    return data, user_data


def create_sparse_adjacency_matrix(data, id_user, num_users, num_items):
    row = []
    col = []
    for i, row_data in zip(id_user, data):
        row.extend([i] * len(row_data))
        col.extend(row_data)
    data = np.ones(len(row))
    adjacency_matrix = coo_matrix((data, (row, col)), shape=(num_users, num_items))
    return adjacency_matrix


def main():
    # 读取测试集和训练集数据
    test_data, test_user = read_file('./ifashion/test.txt')
    train_data, train_user = read_file('./ifashion/train.txt')

    # 计算用户和物品的数量
    num_users_test = max(test_user) + 1
    num_items_test = max(max(row) for row in test_data) + 1
    num_users_train = max(train_user) + 1
    num_items_train = max(max(row) for row in train_data) + 1

    num_users = max(num_users_test, num_users_train)
    num_items = max(num_items_test, num_items_train)

    # 创建测试集和训练集的稀疏邻接矩阵
    test_adj_matrix = create_sparse_adjacency_matrix(test_data, test_user, num_users, num_items)
    train_adj_matrix = create_sparse_adjacency_matrix(train_data, train_user, num_users, num_items)

    return train_adj_matrix, test_adj_matrix

if __name__ == '__main__':
    main()
