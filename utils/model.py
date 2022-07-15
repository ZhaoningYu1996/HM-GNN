import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
        self.dropout = nn.Dropout(0.2)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        # h = self.dropout(h)
        h = self.conv2(graph, h)
        return h

class TwoLayerGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, 512)
        self.conv2 = dglnn.GraphConv(512, 64)
        self.conv3 = dglnn.GraphConv(64, 16)
        self.conv4 = dglnn.GraphConv(16, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.dropout(F.relu(self.conv2(g, x)))
        x = self.dropout(F.relu(self.conv3(g, x)))
        x = F.relu(self.conv4(g, x))
        return x

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g.to(torch.device('cuda:0'))
        self.layers = nn.ModuleList()
        self.fcl_layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=activation, weight=True, norm='both', bias=True))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=activation, weight=True, norm='both', bias=True))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, weight=True, norm='both', bias=True))
        self.dropout = nn.Dropout(p=dropout)
        self.fcl_layers.append(nn.Linear(n_hidden, 16))
        self.fcl_layers.append(nn.Linear(16, n_classes))

    def forward(self, features, norm_edge_weight):
        h = features.to(torch.device('cuda:0'))
        edge_weight = norm_edge_weight.to(torch.device('cuda:0'))
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h, edge_weight=edge_weight)
        return h

class MLPLayer(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 is_cuda):
        super(MLPLayer, self).__init__()
        self.num_layers = num_layers
        self.mlp = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if self.num_layers == 1:
            self.mlp.append(nn.Linear(input_dim, output_dim))
        else:
            self.mlp.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(self.num_layers-2):
                self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(self.num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, h):
        if self.num_layers == 1:
            return self.mlp[0](h)
        else:
            for layer in range(self.num_layers-1):
                h = self.mlp[layer](h)
                h = self.batch_norms[layer](h)
                h = F.relu(h)
        return  self.mlp[self.num_layers-1](h)


class GIN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 drop_rate,
                 learn_eps,
                 neighbor_aggregate_method,
                 final_drop,
                 is_cuda=False):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList()
        self.linear_predictions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.eps_list = nn.Parameter(torch.zeros(self.num_layers-1))
        self.final_drop = final_drop
        self.drop_rate = drop_rate
        self.neighbor_aggregate_method = neighbor_aggregate_method
        self.id_layers = 0
        self.learn_eps = learn_eps

        for layer in range(num_layers-1):
            if layer == 0: 
                self.mlp_layers.append(MLPLayer(num_mlp_layers, input_dim, hidden_dim, hidden_dim, is_cuda))
                self.linear_predictions.append(nn.Linear(input_dim, output_dim))
            else:
                self.mlp_layers.append(MLPLayer(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, is_cuda))
                self.linear_predictions.append(nn.Linear(hidden_dim, output_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linear_predictions.append(nn.Linear(hidden_dim, output_dim))

    def self_eps_aggregate(self, h_v, h_u):
        if self.learn_eps:
            h = (1 + self.eps_list[self.id_layers]) * h_v + h_u
        else:
            h = h_v + h_u
        return h

    @staticmethod
    def message_function(e):
        h = e.data['w'].float() * e.src['h'].float().t()
        h = h.t()
        return {'msg_h': h}

    def reduce_sum_function(self, n):
        h = torch.sum(n.mailbox['msg_h'], dim=1)
        h = self.self_eps_aggregate(n.data['h'], h)
        return {'h': h}

    def node_pooling(self, g):
        if self.neighbor_aggregate_method == 'sum':
            g.update_all(self.message_function, self.reduce_sum_function)

        return g.ndata.pop('h')

    def forward(self, g):
        score_over_layer = 0
        h = g.ndata['feat']

        for layer in range(self.num_layers):
            self.id_layers = layer
            g.ndata['h'] = h

            h = self.node_pooling(g)

            h = self.mlp_layers[layer](h)
            h = self.batch_norms[layer](h)
            h = F.relu(h)

