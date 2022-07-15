from numpy import NaN
import torch
import dgl
import pickle
from ogb.graphproppred import GraphPropPredDataset
import networkx as nx
from utils.data_loader import GenData
from utils.ops_ogb import GenGraph
from tqdm import tqdm
import numpy as np

def gen_features_labels(g, num_cliques):
    features_c = torch.eye(num_cliques)
    num_moles = g.num_nodes() - num_cliques
    features_m = torch.zeros(num_moles, num_cliques)
    for i in tqdm(range(len(g.edges()[0])), desc='Gen Features', unit='feat'):
        if g.edges()[0][i] < num_moles:
            features_m[g.edges()[0][i]][g.edges()[1][i] - num_moles] = 1
        else:
            if g.edges()[1][i] < num_moles:
                features_m[g.edges()[1][i]][g.edges()[0][i] - num_moles] = 1
    features = torch.cat((features_m, features_c), 0)
    return features


dataset = GraphPropPredDataset(name='ogbg-molpcba')
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
graphs = []
labels = []
graph_node_labels = []
node_features = []
edge_features = []

for i in range(len(dataset)):
    g, l = dataset[i]
    labels.append(l)
    edge_feature = []
    n_labels = []
    n_features = []
    [rows, cols] = g['node_feat'].shape
    for j in range(rows):
        n_labels.append(g['node_feat'][j][0].item())
        # n_features.append(g['node_feat'][j][1:])
    graph_node_labels.append(tuple(n_labels))
    for j in range(len(g['edge_index'][0])):
        # index = [g['edge_index'][0][j].item(), g['edge_index'][1][j].item(), 1]
        index = [g['edge_index'][0][j].item(), g['edge_index'][1][j].item(), g['edge_feat'][j][0]]
        edge_feature.append(tuple(index))
    g_nx = nx.Graph()
    g_nx.add_weighted_edges_from(edge_feature)
    graphs.append(g_nx)
data = GenData(graphs, graph_node_labels, labels)
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
u, v = torch.tensor(srn), torch.tensor(dtn)
g_dgl = dgl.graph((u, v))
g_dgl.edata['weight'] = torch.tensor(wte, dtype=torch.float32)

features = gen_features_labels(g_dgl, graph.num_cliques)
g_dgl.ndata['feat'] = features
in_feats = features.size()[1]

train_mask = [True if x in train_idx else False for x in range(int(g_dgl.num_nodes()))]
train_mask = np.array(train_mask)
valid_mask = [True if x in valid_idx else False for x in range(int(g_dgl.num_nodes()))]
valid_mask = np.array(valid_mask)
test_mask = [True if x in test_idx else False for x in range(int(g_dgl.num_nodes()))]
test_mask = np.array(test_mask)
g_dgl.ndata['train_mask'] = torch.from_numpy(train_mask)
g_dgl.ndata['val_mask'] = torch.from_numpy(valid_mask)
g_dgl.ndata['test_mask'] = torch.from_numpy(test_mask)

# normalization
degs = g_dgl.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g_dgl.ndata['norm'] = norm.unsqueeze(1)

with open('ogb-molpcba', 'wb') as save_file:
    pickle.dump(g_dgl, save_file)

