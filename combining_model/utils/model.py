import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, adj, features):
        output = torch.mm(adj, features)
        output = self.linear(output)
        return output


class GCNCOM(nn.Module):
    def __init__(self, input_size, output_size, DEVICE):
        super(GCNCOM, self).__init__()
        self.gcn = GCN(input_size=input_size).to(DEVICE)
        self.fcl_1 = FCL(output_size=output_size).to(DEVICE)
        self.fcl_2 = FCL(output_size=output_size).to(DEVICE)

    def forward(self, adj, features, mask_1, mask_2):
        gcn = self.gcn(adj, features)
        data_1 = gcn[mask_1]
        data_2 = gcn[mask_2]
        output_1 = self.fcl_1(data_1)
        output_2 = self.fcl_2(data_2)
        return output_1, output_2



class GCN(nn.Module):
    def __init__(self, input_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, 512)
        self.gcn2 = GraphConvolution(512, 128)
        self.gcn3 = GraphConvolution(128, 64)
        # self.gcn4 = GraphConvolution(64, 48)
        self.dropout = nn.Dropout(0.2)
        self.fcl1 = nn.Linear(64, 16)
        self.fcl2 = nn.Linear(16, 2)


    def forward(self, adj, features):
        output = F.relu(self.gcn1(adj, features))
        # output = self.gcn3(adj, output)
        output = self.dropout(output)
        output = F.relu(self.gcn2(adj, output))
        output = self.dropout(output)
        output = F.relu(self.gcn3(adj, output))
        output = self.dropout(output)
        # output = F.relu(self.gcn4(adj, output))
        # output = self.dropout(output)
        # output = F.relu(self.fcl1(output))
        # output = self.dropout(output)
        # output = self.fcl2(output)
        return output

class FCL(nn.Module):
    def __init__(self, output_size):
        super(FCL, self).__init__()
        self.fcl1 = nn.Linear(64, 16)
        self.fcl2 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features):
        output = F.relu(self.fcl1(features))
        output = self.dropout(output)
        output = self.fcl2(output)
        return output