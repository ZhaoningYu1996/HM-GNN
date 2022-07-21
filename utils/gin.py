import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from utils.graphcnn import GraphCNN, GraphCNNode


class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


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


class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = False

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(final_dropout)

    def forward(self, g, h, edge_weight):

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h, edge_weight)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if i != 0:
                h = self.drop(h)
        return h

class StochasticGIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type):
        super(StochasticGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = False

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
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
        self.fcl2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, output):
        output = self.dropout(output)
        # output = self.dropout(output)
        output = self.fcl(output)
        
        # output = self.fcl2(output)
        return output

class FCL2(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCL2, self).__init__()
        self.fcl1 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.9)

    def forward(self, output):
        output = self.dropout(output)
        output = self.fcl1(output)
        return output

class TwoGINTorch(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, pre_input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type, graph_pooling_type):
        super(TwoGINTorch, self).__init__()
        self.device = torch.device('cuda:0')
        self.hidden_dim = hidden_dim
        self.graphcnn = GraphCNN(5, num_mlp_layers, pre_input_dim, 64, 64, 0.5, learn_eps, graph_pooling_type,
                                 neighbor_pooling_type, self.device).to(self.device)
        self.gin = GraphCNNode(num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type).to(self.device)

class TwoGIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, pre_input_dim, hidden_dim,
                 output_dim, final_dropout, dropout_0, learn_eps,
                 neighbor_pooling_type, graph_pooling_type):
        super(TwoGIN, self).__init__()
        self.device = torch.device('cuda:0')
        self.hidden_dim = hidden_dim
        self.gin = StochasticGIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.graphcnn = GraphCNN(5, num_mlp_layers, pre_input_dim, 16, 16, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.fcl = FCL(hidden_dim+16, output_dim).to(self.device)

    def forward(self, g, h, edge_weight, graph, num_cliques):
        pre_h = self.graphcnn(graph)
        h = self.gin(g, h, edge_weight)
        h = torch.cat((h, pre_h), 1)
        h = self.fcl(h)
        return h

class ThreeGIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, pre_input_dim1, pre_input_dim2, hidden_dim,
                 output_dim, final_dropout, dropout_0, learn_eps,
                 neighbor_pooling_type, graph_pooling_type):
        super(ThreeGIN, self).__init__()
        self.device = torch.device('cuda:0')
        self.hidden_dim = hidden_dim
        self.gin1 = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.gin2 = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.gin3 = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.graphcnn1 = GraphCNN(5, num_mlp_layers, pre_input_dim1, 16, 16, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.graphcnn2 = GraphCNN(5, num_mlp_layers, pre_input_dim2, 16, 16, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.fcl = FCL(hidden_dim+16, output_dim).to(self.device)

    def forward(self, g, h, edge_weight, graph1, graph2, mask1, mask2):
        pre_h1 = self.graphcnn1(graph1)
        pre_h2 = self.graphcnn2(graph2)
        h = self.gin1(g, h, edge_weight)
        h1 = torch.cat((h[mask1], pre_h1), 1)
        h1 = self.fcl(h1)
        h2 = torch.cat((h[mask2], pre_h2), 1)
        h2 = self.fcl(h2)
        return h1, h2

class GINGraph(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(GINGraph, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer
