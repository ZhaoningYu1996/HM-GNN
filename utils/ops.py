import networkx as nx
import torch
from tqdm import tqdm
import math
import numpy as np
import copy
import gc

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

class GenGraph(object):
    def __init__(self, data):
        self.data = data
        self.nodes_labels = data.node_labels
        self.vocab = {}
        self.whole_node_count = {}
        self.weight_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        g = self.gen_components()
        g = self.update_weight(g)
        # g = self.add_edge(g)
        self.g_final = self.drop_node(g)
        self.num_cliques = self.g_final.number_of_nodes() - len(self.data.g_list)
        del g, self.vocab, self.data, self.whole_node_count, self.weight_vocab, self.node_count,self.edge_count
        gc.collect()


    def gen_components(self):
        g_list = self.data.g_list
        h_g = nx.Graph()
        for g in tqdm(range(len(g_list)), desc='Gen Components', unit='graph'):
            clique_list = []
            mcb = nx.cycle_basis(g_list[g])
            mcb_tuple = [tuple(ele) for ele in mcb]

            edges = []
            for e in g_list[g].edges():
                count = 0
                for c in mcb_tuple:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            edges = list(set(edges))

            for e in edges:
                weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                edge = ((self.nodes_labels[e[0]-1], self.nodes_labels[e[1]-1]), weight)
                clique_id = self.add_to_vocab(edge)
                clique_list.append(clique_id)
                if clique_id not in self.whole_node_count:
                    self.whole_node_count[clique_id] = 1
                else:
                    self.whole_node_count[clique_id] += 1

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g_list[g]))
                ring = []
                for i in range(len(m)):
                    ring.append(self.nodes_labels[m[i]-1])
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
                clique_list.append(cycle_id)
                if cycle_id not in self.whole_node_count:
                    self.whole_node_count[cycle_id] = 1
                else:
                    self.whole_node_count[cycle_id] += 1

            for e in clique_list:
                self.add_weight(e, g)

            c_list = tuple(set(clique_list))
            for e in c_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    self.node_count[e] += 1

            for e in c_list:
                h_g.add_edge(g, e + len(g_list), weight=(self.weight_vocab[(g, e)]/(len(edges) + len(mcb_tuple))))

            for e in range(len(edges)):
                for i in range(e+1, len(edges)):
                    for j in edges[e]:
                        if j in edges[i]:
                            weight = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge = ((self.nodes_labels[edges[e][0] - 1], self.nodes_labels[edges[e][1] - 1]), weight)
                            weight_i = g_list[g].get_edge_data(edges[i][0], edges[i][1])['weight']
                            edge_i = ((self.nodes_labels[edges[i][0] - 1], self.nodes_labels[edges[i][1] - 1]), weight_i)
                            final_edge = tuple(sorted((self.add_to_vocab(edge), self.add_to_vocab(edge_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
            for m in range(len(mcb_tuple)):
                for i in range(m+1, len(mcb_tuple)):
                    for j in mcb_tuple[m]:
                        if j in mcb_tuple[i]:
                            weight = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring = []
                            for t in range(len(mcb_tuple[m])):
                                ring.append(self.nodes_labels[mcb_tuple[m][t] - 1])
                            cycle = (tuple(ring), weight)

                            weight_i = tuple(self.find_ring_weights(mcb_tuple[i], g_list[g]))
                            ring_i = []
                            for t in range(len(mcb_tuple[i])):
                                ring_i.append(self.nodes_labels[mcb_tuple[i][t] - 1])
                            cycle_i = (tuple(ring_i), weight_i)

                            final_edge = tuple(sorted((self.add_to_vocab(cycle), self.add_to_vocab(cycle_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

            for e in range(len(edges)):
                for m in range(len(mcb_tuple)):
                    for i in edges[e]:
                        if i in mcb_tuple[m]:
                            weight_e = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge_e = ((self.nodes_labels[edges[e][0] - 1], self.nodes_labels[edges[e][1] - 1]), weight_e)
                            weight_m = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring_m = []
                            for t in range(len(mcb_tuple[m])):
                                ring_m.append(self.nodes_labels[mcb_tuple[m][t] - 1])
                            cycle_m = (tuple(ring_m), weight_m)

                            final_edge = tuple(sorted((self.add_to_vocab(edge_e), self.add_to_vocab(cycle_m))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

        return h_g

    def add_to_vocab(self, clique):
        c = copy.deepcopy(clique[0])
        weight = copy.deepcopy(clique[1])
        for i in range(len(c)):
            if (c, weight) in self.vocab:
                return self.vocab[(c, weight)]
            else:
                c = self.shift_right(c)
                weight = self.shift_right(weight)
        self.vocab[(c, weight)] = len(list(self.vocab.keys()))
        return self.vocab[(c, weight)]

    def add_weight(self, node_id, g):
        if (g, node_id) not in self.weight_vocab:
            self.weight_vocab[(g, node_id)] = 1
        else:
            self.weight_vocab[(g, node_id)] += 1

    def update_weight(self, g):
        for (u, v) in g.edges():
            if u < len(self.data.g_list):
                g[u][v]['weight'] = g[u][v]['weight'] * (math.log((len(self.data.g_list) + 1) / self.node_count[v - len(self.data.g_list)]))
            else:
                g[u][v]['weight'] = g[u][v]['weight'] * (
                    math.log((len(self.data.g_list) + 1) / self.node_count[u - len(self.data.g_list)]))
        return g

    def add_edge(self, g):
        edges = list(self.edge_count.keys())
        for i in edges:
            g.add_edge(i[0] + len(self.data.g_list), i[1] + len(self.data.g_list), weight=math.exp(math.log(self.edge_count[i] / math.sqrt(self.whole_node_count[i[0]] * self.whole_node_count[i[1]]))))
        return g

    def drop_node(self, g):
        rank_list = []
        node_list = []
        sub_node_list = []
        for v in sorted(g.nodes()):
            if v > len(self.data.g_list):
                rank_list.append(self.node_count[v - len(self.data.g_list)] / len(self.data.g_list))
                node_list.append(v)
        sorted_list = sorted(rank_list)
        a = int(len(sorted_list) * 0.9)
        threshold_num = sorted_list[a]
        for i in range(len(rank_list)):
            if rank_list[i] > threshold_num:
                sub_node_list.append(node_list[i])
        self.removed_nodes = sub_node_list
        count = 0
        label_mapping = {}
        for v in sorted(g.nodes()):
            if v in sub_node_list:
                count += 1
            else:
                label_mapping[v] = v - count
        for v in sub_node_list:
            g.remove_node(v)
        
        g = nx.relabel_nodes(g, label_mapping)
        return g

    @staticmethod
    def shift_right(l):
        if type(l) == int:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return tuple([l[-1]] + l[:-1])
        else:
            print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g):
        weight_list = []
        for i in range(len(ring)-1):
            weight = g.get_edge_data(ring[i], ring[i+1])['weight']
            weight_list.append(weight)
        weight = g.get_edge_data(ring[-1], ring[0])['weight']
        weight_list.append(weight)
        return weight_list

