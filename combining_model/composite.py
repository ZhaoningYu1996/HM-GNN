import argparse
import random
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import FileLoader
from utils.ops import Generator, norm
import copy
import numpy as np
from tqdm import tqdm
from utils.model import GCN, FCL, GCNCOM
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


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=0, help='seed')
    parser.add_argument('-data', default='PTC', help='data folder name')
    parser.add_argument('-num_epochs', type=int, default=1450, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
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


def shift_right(l):
    if type(l) == int:
        return l
    elif type(l) == tuple:
        l = list(l)
        return tuple([l[-1]] + l[:-1])
    elif type(l) == list:
        return [l[-1]] + l[:-1]


def add_to_vocab(clique, vocab):
    c = copy.deepcopy(clique[0])
    weight = copy.deepcopy(clique[1])
    for i in range(len(c)):
        if (c, weight) in vocab:
            return vocab[(c, weight)]
        else:
            c = shift_right(c)
            weight = shift_right(weight)
    vocab[(c, weight)] = len(vocab) + 1
    return vocab[(c, weight)]


def gen_feature_adj_labels(g1, g2, labels_1, labels_2, num_cliques):
    number_of_cliques = num_cliques
    number_of_moles_1 = len(labels_1)
    number_of_moles_2 = len(labels_2)
    feature_c = np.eye(number_of_cliques, dtype=np.float32)
    feature_m_1 = np.zeros((number_of_moles_1, number_of_cliques))
    feature_m_2 = np.zeros((number_of_moles_2, number_of_cliques))
    for i in tqdm(range(1, number_of_moles_1+1), desc="feature matrix", unit="graph"):
        if i == 11158:
            print('M{}'.format(i))
        if 'M{}'.format(i) in g1.nodes:
            if len(g1.adj["M{}".format(i)]) > 0:
                for k, dict in g1.adj["M{}".format(i)].items():
                    feature_m_1[i-1][int(k[1:])-1] = 1

    for i in range(1, number_of_moles_2+1):
        if len(g2.adj['M{}'.format(i)]) > 0:
            for k, dict in g2.adj['M{}'.format(i)].items():
                feature_m_2[i-1][int(k[1:])-1] = 1

    features = np.r_[feature_c, feature_m_1]
    new_features = np.r_[features, feature_m_2]
    # features = np.concatenate((feature_c, feature_m), axis=1)

    # Generate node labels
    y = np.empty(len(labels_1)+len(labels_2)+number_of_cliques, int)
    for i in range(number_of_cliques, len(y)):
        if i < number_of_cliques + len(labels_1):
            y[i] = labels_1[i - number_of_cliques]
        else:
            y[i] = labels_2[i - number_of_cliques - len(labels_1)]
    y_1 = np.empty(len(labels_1)+number_of_cliques, int)
    for i in range(number_of_cliques, len(y_1)):
        y_1[i] = labels_1[i - number_of_cliques]
    y_2 = np.empty(len(labels_2)+number_of_cliques, int)
    for i in range(number_of_cliques, len(y_2)):
        y_2[i] = labels_2[i - number_of_cliques]

    adj = np.zeros((len(labels_1)+len(labels_2)+number_of_cliques, len(labels_1)+len(labels_2)+number_of_cliques))
    for k, v in g1.adj.items():
        for item in v.keys():
            if k[0] == "M":
                if item[0] == "M":
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = g1.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = g1.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
    for k, v in g2.adj.items():
        for item in v.keys():
            if k[0] == 'M':
                if item[0] == 'M':
                    adj[int(k[1:]) + number_of_cliques + len(labels_1) - 1][int(item[1:]) + number_of_cliques + len(labels_1) - 1]  = g2.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) + number_of_cliques + len(labels_1) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques + len(labels_1) - 1] = g2.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
    adj = norm(adj)
    return new_features, adj, y_1, y_2


def gen_feature_adj_labels_no_test(g1, g2, labels_1, labels_2, num_cliques, train_idx_2):
    number_of_moles_1 = len(labels_1)
    number_of_moles_2 = len(labels_2)
    train_idx_2 = sorted(train_idx_2)
    feature_c = np.eye(num_cliques, dtype=np.float32)
    feature_m_1 = np.zeros((number_of_moles_1, num_cliques))
    feature_m_2_test = np.zeros((number_of_moles_2 - len(train_idx_2), num_cliques))
    feature_m_2_train = np.zeros((len(train_idx_2), num_cliques))
    count = 0
    for i in range(1, number_of_moles_1+1):
        for k, dict in g1.adj["M{}".format(i)].items():
            feature_m_1[i-1][int(k[1:])-1] = 1
    for i in range(1, number_of_moles_2 + 1):
        if i-1 in train_idx_2:
            count += 1
            continue
        for k, dict in g2.adj['M{}'.format(i)].items():
            feature_m_2_test[i - count - 1][int(k[1:]) - 1] = 1
    for i in range(len(train_idx_2)):
        for k, dict in g2.adj['M{}'.format(train_idx_2[i]+1)].items():
            feature_m_2_train[i][int(k[1:]) - 1] = 1
    features_train = np.r_[feature_c, feature_m_1]
    new_features_train = np.r_[features_train, feature_m_2_train]
    new_features_test = np.r_[feature_c, feature_m_2_test]

    y_train = np.empty(num_cliques + len(labels_1) + len(train_idx_2), int)
    y_test = np.empty(num_cliques + len(labels_2) - len(train_idx_2), int)
    for i in range(num_cliques, len(y_train)):
        if i < num_cliques + len(labels_1):
            y_train[i] = labels_1[i - num_cliques]
        else:
            y_train[i] = labels_2[train_idx_2[i - num_cliques - len(labels_1)]]
    ct = 0
    for j in range(len(labels_2)):
        if j in train_idx_2:
            ct += 1
            continue
        else:
            y_test[j+num_cliques-ct] = labels_2[j]

    adj_test = np.zeros((len(labels_2) + num_cliques - len(train_idx_2), len(labels_2) + num_cliques - len(train_idx_2)))
    adj_train = np.zeros((len(labels_1) + len(train_idx_2) + num_cliques, len(labels_1) + len(train_idx_2) + num_cliques))
    for k, v in g1.adj.items():
        for item in v.keys():
            if k[0] == "M":
                if item[0] == "M":
                    adj_train[int(k[1:]) + num_cliques - 1][int(item[1:]) + num_cliques - 1] = g1.get_edge_data(k, item)['weight']
                else:
                    adj_train[int(k[1:]) + num_cliques - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj_train[int(k[1:]) - 1][int(item[1:]) + num_cliques - 1] = g1.get_edge_data(k, item)['weight']
                else:
                    adj_train[int(k[1:]) - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
    for k, v in g2.adj.items():
        for item in v.keys():
            if k[0] == 'M':
                if int(k[1:]) - 1 in train_idx_2:
                    if item[0] == 'M':
                        adj_train[train_idx_2.index(int(k[1:]) - 1) + num_cliques + len(labels_1)][train_idx_2.index(int(item[1:]) - 1) + num_cliques + len(labels_1)] = g2.get_edge_data(k, item)['weight']
                    else:
                        adj_train[train_idx_2.index(int(k[1:]) - 1) + num_cliques + len(labels_1)][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
                else:
                    if item[0] == 'C':
                        if int(k[1:]) - 1 < train_idx_2[0]:
                            adj_test[int(k[1:]) - 1 + num_cliques][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
                        elif int(k[1:]) - 1 > train_idx_2[-1]:
                            adj_test[int(k[1:]) - 1 + num_cliques - len(train_idx_2)][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
                        else:
                            for t in range(len(train_idx_2) - 1):
                                if int(k[1:]) - 1 < train_idx_2[t+1] and int(k[1:]) - 1 > train_idx_2[t]:
                                    adj_test[int(k[1:]) - 1 + num_cliques - t - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    if int(item[1:]) - 1 in train_idx_2:
                        adj_train[int(k[1:]) - 1][train_idx_2.index(int(item[1:]) - 1) + num_cliques + len(labels_1)] = g2.get_edge_data(k, item)['weight']
                    else:
                        if int(item[1:]) - 1 < train_idx_2[0]:
                            adj_test[int(k[1:]) - 1][int(item[1:]) - 1 + num_cliques] = g2.get_edge_data(k, item)['weight']
                        elif int(item[1:]) - 1 > train_idx_2[-1]:
                            adj_test[int(k[1:]) - 1][int(item[1:]) - 1 + num_cliques - len(train_idx_2)] = g2.get_edge_data(k, item)['weight']
                        else:
                            for t in range(len(train_idx_2) - 1):
                                if int(item[1:]) - 1 < train_idx_2[t + 1] and int(item[1:]) - 1 > train_idx_2[t]:
                                    adj_test[int(k[1:]) - 1][int(item[1:]) - 1 + num_cliques - t - 1] = g2.get_edge_data(k, item)['weight']
                else:
                    adj_train[int(k[1:]) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
                    adj_test[int(k[1:]) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
    adj_test = norm(adj_test)
    adj_train = norm(adj_train)
    return new_features_test, new_features_train, adj_test, adj_train, y_test, y_train


def gen_feature_adj_labels_two(g1, g2, labels_1, labels_2, num_cliques, train_idx_2):
    number_of_cliques = num_cliques
    number_of_moles_1 = len(labels_1)
    number_of_moles_2 = len(labels_2)
    feature_c = np.eye(number_of_cliques, dtype=np.float32)
    feature_m_1 = np.zeros((number_of_moles_1, number_of_cliques))
    feature_m_2 = np.zeros((number_of_moles_2, number_of_cliques))
    feature_m_2_train = np.zeros((number_of_moles_2, number_of_cliques))
    for i in tqdm(range(1, number_of_moles_1+1), desc="feature matrix", unit="graph"):
        if i == 11158:
            print('M{}'.format(i))
        if 'M{}'.format(i) in g1.nodes:
            if len(g1.adj["M{}".format(i)]) > 0:
                for k, dict in g1.adj["M{}".format(i)].items():
                    feature_m_1[i-1][int(k[1:])-1] = 1

    for i in range(1, number_of_moles_2+1):
        if len(g2.adj['M{}'.format(i)]) > 0:
            for k, dict in g2.adj['M{}'.format(i)].items():
                feature_m_2[i-1][int(k[1:])-1] = 1

    for i in range(1, number_of_moles_2+1):
        if i in train_idx_2:
            for k, dict in g2.adj['M{}'.format(i)].items():
                feature_m_2_train[i-1][int(k[1:])-1] = 1
    features_train = np.r_[feature_c, feature_m_1]
    new_features_train = np.r_[features_train, feature_m_2_train]

    features = np.r_[feature_c, feature_m_1]
    new_features = np.r_[features, feature_m_2]

    # Generate node labels
    y = np.empty(len(labels_1)+len(labels_2)+number_of_cliques, int)
    print("length y", len(y))
    for i in range(number_of_cliques, len(y)):
        if i < number_of_cliques + len(labels_1):
            y[i] = labels_1[i - number_of_cliques]
        else:
            y[i] = labels_2[i - number_of_cliques - len(labels_1)]

    adj = np.zeros((len(labels_1)+len(labels_2)+number_of_cliques, len(labels_1)+len(labels_2)+number_of_cliques))
    adj_train = np.zeros((len(labels_1)+len(labels_2)+number_of_cliques, len(labels_1)+len(labels_2)+number_of_cliques))
    for k, v in g1.adj.items():
        for item in v.keys():
            if k[0] == "M":
                if item[0] == "M":
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = g1.get_edge_data(k, item)['weight']
                    adj_train[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = g1.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
                    adj_train[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = g1.get_edge_data(k, item)['weight']
                    adj_train[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = g1.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
                    adj_train[int(k[1:]) - 1][int(item[1:]) - 1] = g1.get_edge_data(k, item)['weight']
    for k, v in g2.adj.items():
        for item in v.keys():
            if k[0] == 'M':
                if item[0] == 'M':
                    adj[int(k[1:]) + number_of_cliques + len(labels_1) - 1][int(item[1:]) + number_of_cliques + len(labels_1) - 1]  = g2.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) + number_of_cliques + len(labels_1) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques + len(labels_1) - 1] = g2.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
    for k, v in g2.adj.items():
        if k not in train_idx_2:
            continue
        for item in v.keys():
            if k[0] == 'M':
                if item[0] == 'M':
                    adj_train[int(k[1:]) + number_of_cliques + len(labels_1) - 1][
                        int(item[1:]) + number_of_cliques + len(labels_1) - 1] = g2.get_edge_data(k, item)['weight']
                else:
                    adj_train[int(k[1:]) + number_of_cliques + len(labels_1) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj_train[int(k[1:]) - 1][int(item[1:]) + number_of_cliques + len(labels_1) - 1] = g2.get_edge_data(k, item)['weight']
                else:
                    adj_train[int(k[1:]) - 1][int(item[1:]) - 1] = g2.get_edge_data(k, item)['weight']
    adj = norm(adj)
    adj_train = norm(adj_train)
    return new_features, new_features_train, adj, adj_train, y


def sep_data(labels, seed):
    skf = KFold(n_splits=10, shuffle=True)
    labels = labels
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


def train(num_epochs, mask_data_1, mask_data_2, net, optimizer, features, adj, loss, idx_train_1, idx_train_2, idx_test_2, labels_1, labels_2):
    # tacc_mx_1 = 0.0
    tacc_mx = 0.0
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        output_1, output_2 = net(adj, features, mask_data_1, mask_data_2)
        loss_train_1 = loss(output_1[idx_train_1], labels_1[idx_train_1])
        loss_train_2 = loss(output_2[idx_train_2], labels_2[idx_train_2])
        loss_train = 10*loss_train_1 + loss_train_2
        loss_train.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            test_output_1, test_output_2 = net(adj, features, mask_data_1, mask_data_2)
            acc_test_2 = accuracy(test_output_2[idx_test_2], labels_2[idx_test_2])
            if acc_test_2 >= tacc_mx:
                tacc_mx = acc_test_2

    return tacc_mx

def main():
    args = get_args()
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ptc_mr = FileLoader(args).load_data()
    args.data = 'PTC_FR'
    ptc_fr = FileLoader(args).load_data()
    args.data = 'PTC_FM'
    ptc_fm = FileLoader(args).load_data()
    labels_mr = ptc_mr.graph_labels
    labels_fr = ptc_fr.graph_labels
    # labels_fm = ptc_fm.graph_labels
    # print(labels_mr)
    labels_fr_tsne = copy.deepcopy(labels_fr)
    labels_mr_tsne = copy.deepcopy(labels_mr)
    for i in range(len(labels_fr_tsne)):
        if labels_fr_tsne[i] == -1:
            labels_fr_tsne[i] = 2
        if labels_fr_tsne[i] == 1:
            labels_fr_tsne[i] = 3
    for i in range(len(labels_mr_tsne)):
        if labels_mr_tsne[i] == -1:
            labels_mr_tsne[i] = 4
        if labels_mr_tsne[i] == 1:
            labels_mr_tsne[i] = 5
    for i in range(len(labels_mr)):
        if labels_mr[i] == -1:
            labels_mr[i] = 0
    for i in range(len(labels_fr)):
        if labels_fr[i] == -1:
            labels_fr[i] = 0
    g_mr, vocab_mr = Generator(ptc_mr).gen_large_graph()
    g_fr, vocab_fr = Generator(ptc_fr).gen_large_graph()
    keys_mr = list(vocab_mr.keys())
    keys_fr = list(vocab_fr.keys())
    count_mr = 0
    count_mm = 0
    change = {}
    for key in keys_fr:
        new_position = add_to_vocab(key, vocab_mr)
        change["C{}".format(vocab_fr[key])] = "C{}".format(new_position)
    new_g_fr = nx.relabel_nodes(g_fr, change)

    print("length of whole vocab", len(vocab_mr))
    features, adj, node_labels_mr, node_labels_fr = gen_feature_adj_labels(g_mr, new_g_fr, labels_mr, labels_fr, len(vocab_mr))
    features = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    node_labels_mr = torch.tensor(node_labels_mr, dtype=torch.long).to(DEVICE)
    node_labels_fr = torch.tensor(node_labels_fr, dtype=torch.long).to(DEVICE)
    adj = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
    bacc_1 = 0.0
    bacc_2 = 0.0
    bseed_1 = -1
    bseed_2 = -1
    bst_1 = 0.0
    bst_2 = 0.0
    all_mr = []
    all_fr = []
    for i in range(len(labels_mr)):
        all_mr.append(i + len(vocab_mr))
    for i in range(len(labels_fr)):
        all_fr.append(i + len(vocab_mr))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train_idx_mr, test_idx_mr = sep_data(labels_mr, args.seed)
    train_idx_fr, test_idx_fr = sep_data(labels_fr, args.seed)
    test_acc_mr = []
    test_acc_fr = []
    for i in range(10):
        train_mr = train_idx_mr[i]
        test_mr = test_idx_mr[i]
        train_fr = train_idx_fr[i]
        test_fr = test_idx_fr[i]
        for j in range(len(train_mr)):
            train_mr[j] = train_mr[j] + len(vocab_mr)
        for j in range(len(train_fr)):
            train_fr[j] = train_fr[j] + len(vocab_mr)
        for j in range(len(test_mr)):
            test_mr[j] = test_mr[j] + len(vocab_mr)
        for j in range(len(test_fr)):
            test_fr[j] = test_fr[j] + len(vocab_mr)
        mask_train_mr = [True if x in train_mr else False for x in range(len(vocab_mr) + len(labels_mr))]
        mask_train_fr = [True if x in train_fr else False for x in
                            range(len(vocab_mr) + len(labels_fr))]
        mask_all_mr = [True if x in all_mr else False for x in range(len(vocab_mr) + len(labels_mr))]
        mask_all_fr = [True if x in all_fr else False for x in range(len(vocab_mr) + len(labels_fr))]
        mask_test_mr = [True if x in test_mr else False for x in range(len(vocab_mr) + len(labels_mr))]
        mask_test_fr = [True if x in test_fr else False for x in range(len(vocab_mr) + len(labels_fr))]
        mask_mr = [True if x < (len(vocab_mr) + len(labels_mr)) else False for x in range(len(vocab_mr) + len(labels_mr) + len(labels_fr))]
        mask_fr = [True if x < len(vocab_mr) or x >= (len(vocab_mr) + len(labels_mr)) else False for x in
                    range(len(vocab_mr) + len(labels_mr) + len(labels_fr))]

        net_1 = GCNCOM(input_size=len(vocab_mr), output_size=2, DEVICE=DEVICE).to(DEVICE)
        net_2 = GCNCOM(input_size=len(vocab_mr), output_size=2, DEVICE=DEVICE).to(DEVICE)
        optimizer_1 = optim.Adam(net_1.parameters(), lr=args.lr, weight_decay=args.w_d)
        optimizer_2 = optim.Adam(net_2.parameters(), lr=args.lr, weight_decay=args.w_d)
        loss1 = nn.CrossEntropyLoss()
        loss2 = nn.CrossEntropyLoss()
        acc_mr = train(2000, mask_fr, mask_mr, net_1, optimizer_1, features, adj, loss1, mask_all_fr, mask_train_mr, mask_test_mr, node_labels_fr, node_labels_mr)
        acc_mm = train(2000, mask_mr, mask_fr, net_2, optimizer_2, features, adj, loss2, mask_all_mr, mask_train_fr, mask_test_fr, node_labels_mr, node_labels_fr)
        test_acc_mr.append(acc_mr)
        test_acc_fr.append(acc_mm)
    print(f"The accuracy of PTC is {Average(test_acc_mr)}")
    print(f"Standard Deviation of data PTC is {st.stdev(test_acc_mr)}")
    print(f"The accuracy of ptc_fr is {Average(test_acc_fr)}")
    print(f"Standard Deviation of data ptc_fr is {st.stdev(test_acc_fr)}")
    if Average(test_acc_mr) >= bacc_1:
        bacc_1 = Average(test_acc_mr)
        bseed_1 = args.seed
        bst_1 = st.stdev(test_acc_mr)
    if Average(test_acc_fr) >= bacc_2:
        bacc_2 = Average(test_acc_fr)
        bseed_2 = args.seed
        bst_2 = st.stdev(test_acc_fr)
    print(f"The best model setting for PTC is seed{bseed_1}, with accuracy {bacc_1}, and st {bst_1}")
    print(f"The best model setting for PTC is seed{bseed_2}, with accuracy {bacc_2}, and st {bst_2}")

if __name__ == '__main__':
    CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    main()
