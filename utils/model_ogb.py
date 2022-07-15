import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from utils.conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean
from dgl.nn.pytorch.conv import GINConv


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300,
                    gnn_type = 'gin', virtual_node = True, residual = True, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)

        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return h_graph


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class StochasticGIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, first_hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type):
        super(StochasticGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = False

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, first_hidden_dim, first_hidden_dim)
            elif layer == 1:
                mlp = MLP(num_mlp_layers, first_hidden_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            if layer == 0:
                self.batch_norms.append(nn.BatchNorm1d(first_hidden_dim))
            else:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(final_dropout)

    def forward(self, blocks, h, edge_weight):

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](blocks[i], h, edge_weight[i])
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if i != 0:
                h = self.drop(h)
        return h


class FCL(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCL, self).__init__()
        self.fcl = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.7)

    def forward(self, gcn_output):
        output = self.fcl(gcn_output)
        return output

class TwoGraphGCN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, first_hididen_dim, first_graph_hidden_dim,
                 output_dim, final_dropout, dropout_g, learn_eps,
                 neighbor_pooling_type, device):
        super(TwoGraphGCN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.gin = GNN(num_tasks=1, num_layer=5, emb_dim=first_graph_hidden_dim,
                    drop_ratio=dropout_g, virtual_node=True).to(device)
        self.gin2 = StochasticGIN(num_layers, num_mlp_layers, input_dim, hidden_dim, first_hididen_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.fcl = FCL(hidden_dim + first_graph_hidden_dim, output_dim).to(self.device)
        self.mlp = MLPReadout(hidden_dim + first_graph_hidden_dim, 1).to(self.device)

    def forward(self, g, h, edge_weight, graph):
        pre_h = self.gin(graph)
        h = self.gin2(g, h, edge_weight)
        h = torch.cat((h, pre_h), 1)
        h = self.mlp(h)
        return h

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, decreasing_dim=True):
        super().__init__()
        if decreasing_dim:
            list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
            list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        else:
            list_FC_layers = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
            list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y