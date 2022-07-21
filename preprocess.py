from networkx.generators.random_graphs import barabasi_albert_graph
from tqdm import tqdm
import pickle
import argparse
import dgl
import dgl.data
from dgl.nn.pytorch import EdgeWeightNorm

import torch
from utils.data_loader import FileLoader
from utils.ops import GenGraph
from utils.ops import load_data


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-data', default='NCI1', help='data folder name')
    parser.add_argument('-edge_weight', type=bool, default=False, help='If data have edge labels')
    args, _ = parser.parse_known_args()
    return args

def gen_features_labels(labels, g, num_cliques):
    emtpy_labels = torch.zeros(num_cliques, dtype=torch.long)
    labels = torch.cat((labels, emtpy_labels), 0)
    features_c = torch.eye(num_cliques)
    num_moles = g.num_nodes() - num_cliques
    features_m = torch.zeros(num_moles, num_cliques)
    for i in tqdm(range(len(g.edges()[0])), desc='Gen feature', unit='edge'):
        if g.edges()[0][i] < num_moles:
            features_m[g.edges()[0][i]][g.edges()[1][i] - num_moles] = 1
        else:
            if g.edges()[1][i] < num_moles:
                features_m[g.edges()[1][i]][g.edges()[0][i] - num_moles] = 1
    features = torch.cat((features_m, features_c), 0)
    return features, labels

args = get_args()
data = FileLoader(args).load_data()
graph = GenGraph(data)
num_cliques = graph.num_cliques
print('Number of cliques:', num_cliques)
edge_list = list(graph.g_final.edges())

srn = []
dtn = []
wte = []
for i in edge_list:
    srn.append(i[0])
    srn.append((i[1]))
    dtn.append(i[1])
    dtn.append(i[0])
    wte.append(graph.g_final.get_edge_data(i[0], i[1])['weight'])
    wte.append(graph.g_final.get_edge_data(i[1], i[0])['weight'])
    # wte.append(1)
    # wte.append(1)
u, v = torch.tensor(srn), torch.tensor(dtn)
g_dgl = dgl.graph((u, v))
g_dgl.edata['weight'] = torch.tensor(wte, dtype=torch.float32)

# Change graph labels to 0 and 1
graph_labels = []
for i in range(len(data.graph_labels)):
    # if data.graph_labels[i] == -1:
    #     graph_labels.append(0)
    # else:
    #     graph_labels.append(1)
    # graph_labels.append(data.graph_labels[i] - 1)
    graph_labels.append(data.graph_labels[i])
graph_labels = torch.tensor(graph_labels, dtype=torch.long)
features, labels = gen_features_labels(graph_labels, g_dgl, graph.num_cliques)

g_dgl.ndata['feat'] = features
g_dgl.ndata['labels'] = labels
in_feats = features.size()[1]
edge_weight = g_dgl.edata['weight']

norm = EdgeWeightNorm(norm='both')
norm_edge_weight = norm(g_dgl, edge_weight)
edge_weight = norm_edge_weight
g_dgl.edata['edge_weight'] = edge_weight

degs = g_dgl.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g_dgl.ndata['norm'] = norm.unsqueeze(1)
print(g_dgl.edges()[0].size())

with open('preprocessed_datasets/' + args.data, 'wb') as save_file:
    pickle.dump(g_dgl, save_file)