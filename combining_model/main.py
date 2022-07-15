import argparse
import time
import random
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import FileLoader
from utils.ops import Generator, norm
import numpy as np
from tqdm import tqdm
from utils.model import GCN
import matplotlib.pyplot as plt
import statistics as st
from sklearn.model_selection import StratifiedKFold, KFold
import json
import time
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=0, help='seed')
    parser.add_argument('-data', default='PTC_MM', help='data folder name')
    parser.add_argument('-num_epochs', type=int, default=15, help='epochs')
    parser.add_argument('-batch', type=int, default=0, help='batch size')
    parser.add_argument('-edge_weight', type=bool, default=True, help='If data have edge labels')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-w_d', type=float, default=0.0005, help='weight decay')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0., help='drop net')
    parser.add_argument('-drop_c', type=float, default=0., help='drop output')
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    parser.add_argument('-learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('-learning_rate_decay_factor', type=float, default=0.8)
    args, _ = parser.parse_known_args()
    return args


def gen_feature(g_graph, labels, num_cliques):
    number_of_moles = len(labels)
    feature = np.zeros((number_of_moles, num_cliques))
    for i in range(1, number_of_moles+1):
        for k, dict in g_graph.adj["M{}".format(i)].items():
            feature[i-1][int(k[1:])-1] = 1
    return feature

def gen_feature_adj_labels(g_graph, labels, num_cliques):
    number_of_cliques = num_cliques
    number_of_moles = len(labels)
    feature_c = np.eye(number_of_cliques, dtype=np.float32)
    feature_m = np.zeros((number_of_moles, number_of_cliques))
    for i in tqdm(range(1, number_of_moles+1), desc="feature matrix", unit="graph"):
        if i == 11158:
            print('M{}'.format(i))
        if 'M{}'.format(i) in g_graph.nodes:
            if len(g_graph.adj["M{}".format(i)]) > 0:
                for k, dict in g_graph.adj["M{}".format(i)].items():
                    # if int(k[1:]) != 279 and int(k[1:]) != 259:
                    feature_m[i-1][int(k[1:])-1] = 1

    features = np.r_[feature_c, feature_m]
    # features = np.concatenate((feature_c, feature_m), axis=1)

    # Generate node labels
    y = np.empty(g_graph.number_of_nodes(), int)
    print("length y", len(y))
    for i in range(number_of_cliques, len(y)):
        # print("i", i)
        # print("len feature", len(feature_c))
        y[i] = labels[i - number_of_cliques]

    adj = np.zeros((g_graph.number_of_nodes(), g_graph.number_of_nodes()))
    for k, v in g_graph.adj.items():
        for item in v.keys():
            if k[0] == "M":
                if item[0] == "M":
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = g_graph.get_edge_data(k, item)['weight']
                    # adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = 1
                else:
                    if int(item[1:]) == 279 or int(item[1:]) == 259:
                        continue
                    else:
                        adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
                    # adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = 1
            else:
                # if int(item[1:]) == 279 or int(item[1:]) == 259:
                #     continue
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = g_graph.get_edge_data(k, item)['weight']
                    # adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = 1
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
                    # adj[int(k[1:]) - 1][int(item[1:]) - 1] = 1

    adj = norm(adj)
    return features, adj, y

def gen_feature_adj_labels_ten(g_graph, labels, num_cliques, idx_train):
    number_of_cliques = num_cliques
    number_of_moles = len(labels)
    idx_train = sorted(idx_train)
    feature_c = np.eye(number_of_cliques, dtype=np.float32)
    feature_m = np.zeros((number_of_moles, number_of_cliques))
    feature_m_train = np.zeros((len(idx_train), number_of_cliques))
    for i in tqdm(range(1, number_of_moles+1), desc="feature matrix", unit="graph"):
        if i == 11158:
            print('M{}'.format(i))
        if 'M{}'.format(i) in g_graph.nodes:
            if len(g_graph.adj["M{}".format(i)]) > 0:
                for k, dict in g_graph.adj["M{}".format(i)].items():
                    feature_m[i-1][int(k[1:])-1] = 1
    for i in range(len(idx_train)):
        for k, dict in g_graph.adj['M{}'.format(idx_train[i]+1)].items():
            feature_m_train[i][int(k[1:])-1] = 1


    features = np.r_[feature_c, feature_m]
    features_train = np.r_[feature_c, feature_m_train]
    # features = np.concatenate((feature_c, feature_m), axis=1)

    # Generate node labels
    y = np.empty(g_graph.number_of_nodes(), int)
    y_train = np.empty(len(idx_train)+number_of_cliques, int)
    # print("length y", len(y))
    for i in range(number_of_cliques, len(y)):
        y[i] = labels[i - number_of_cliques]
    for i in range(number_of_cliques, len(y_train)):
        y_train[i] = labels[idx_train[i-number_of_cliques]]

    adj = np.zeros((g_graph.number_of_nodes(), g_graph.number_of_nodes()))
    adj_train = np.zeros((number_of_cliques + len(idx_train), number_of_cliques + len(idx_train)))
    for k, v in g_graph.adj.items():
        for item in v.keys():
            if k[0] == "M":
                if int(k[1:])-1 in idx_train:
                    if item[0] == 'C':
                        adj_train[idx_train.index(int(k[1:])-1) + number_of_cliques][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
                if item[0] == "M":
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = g_graph.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = g_graph.get_edge_data(k, item)['weight']
                    if int(item[1:])-1 in idx_train:
                        adj_train[int(k[1:]) - 1][idx_train.index(int(item[1:])-1) + number_of_cliques] = g_graph.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
                    adj_train[int(k[1:]) - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']

    adj = norm(adj)
    adj_train = norm(adj_train)
    return features, features_train, adj, adj_train, y, y_train

def sep_data(data, seed):
    skf = KFold(n_splits=10, shuffle=True)
    labels = data.graph_labels
    test_idx = []
    train_idx = []
    for train_val_index, test_index in skf.split(np.zeros(len(labels))):
        test_index = list(test_index)
        train_val_index = list(train_val_index)
        test_idx.append(test_index)
        train_idx.append(train_val_index)
    return train_idx, test_idx

def sep_data_three(data, seed):
    skf = KFold(n_splits=10, shuffle=True)
    labels = data.graph_labels
    test_idx = []
    train_idx = []
    for train_val_index, test_index in skf.split(np.zeros(len(labels))):
        test_index = list(test_index)
        train_val_index = list(train_val_index)
        test_idx.append(test_index)
        train_idx.append(train_val_index)
    return train_idx, test_idx

def accuracy(outputs, y):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == y).item() / len(preds)

def Average(lst):
    return sum(lst) / len(lst)

def train_one_epoch(num_epochs, net, optimizer, features, adj, loss, idx_train, idx_test, labels):
    tacc_mx = 0.0
    state_dict_early_model = None
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        output = net(adj, features)
        loss_train = loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            test_output = net(adj, features)
            # print(len(test_output[idx_test]))
            acc_test = accuracy(test_output[idx_test], labels[idx_test])
            if acc_test >= tacc_mx:
                tacc_mx = acc_test

    return state_dict_early_model, tacc_mx

def main():
    args = get_args()
    tacc = 0.0
    bseed = -1
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # DEVICE = 'cpu'
    data = FileLoader(args).load_data()

    labels = data.graph_labels
    for y in range(len(labels)):
        # labels[y] = labels[y] - 1
        if labels[y] == -1:
            labels[y] = 0
        elif labels[y] == +1:
            labels[y] = 1
    g_graph, vocab = Generator(data).gen_large_graph()

    num_cliques = len(vocab)
    print("num_cliques:", num_cliques)
    features, adj, node_labels = gen_feature_adj_labels(g_graph, labels, num_cliques)
    features = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    node_labels = torch.tensor(node_labels, dtype=torch.long).to(DEVICE)
    adj = torch.tensor(adj, dtype=torch.float32).to(DEVICE)

    tacc = 0.0
    std = 0.0
    total_acc = 0.0
    bseed = -1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train_idx, test_idx = sep_data(data, args.seed)
    test_acc = []
    test_loss = []
    idx = []
    for i in range(len(labels)):
        idx.append(i)
    random.shuffle(idx)
    for i in range(10):
        net = GCN(input_size=num_cliques).to(DEVICE)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.w_d)
        loss = nn.CrossEntropyLoss()
        train = train_idx[i]
        train_indices = [i+num_cliques for i in train]
        test = test_idx[i]
        test_indices = [i+num_cliques for i in test]
        # print("val and test", val_indices, test_indices)
        mask_train = [True if x in train_indices else False for x in range(num_cliques + len(labels))]
        mask_test = [True if x in test_indices else False for x in range(num_cliques + len(labels))]

        state_dict, acc = train_one_epoch(2000, net, optimizer, features, adj, loss, mask_train, mask_test, node_labels)
        test_acc.append(acc)

    print(f"The total val accuracy of model with seed {args.seed} is {Average(test_acc)}")
    print(f"Standard Deviation of model is {st.stdev(test_acc)}")


if __name__ == '__main__':
    CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    main()


