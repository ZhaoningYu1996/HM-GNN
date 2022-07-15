import networkx as nx
from tqdm import tqdm


class G_data(object):
    def __init__(self, g_list, node_labels, graph_labels):
        self.g_list = g_list
        self.node_labels = node_labels
        self.graph_labels = graph_labels


class FileLoader(object):
    def __init__(self, args):
        self.args = args


    def load_data(self):
        data = self.args.data
        print(data)
        with open('/home/znyu/projects/Mole-GCN/src/data/%s/A.txt' % (data), 'r') as f:
            edges = f.read().splitlines()

        edges = [tuple(map(int, e.replace(" ", "").split(","))) for e in edges]  # Convert string edges into int list
        print("edges", len(edges))

        with open('/home/znyu/projects/Mole-GCN/src/data/%s/graph_indicator.txt' % (data), 'r') as f:
            g = f.readlines()
        g = [int(i) for i in g]  # Graph indicator for nodes
        print("g", len(g))

        weights = []
        if self.args.edge_weight:
            with open('/home/znyu/projects/Mole-GCN/src/data/%s/edge_labels.txt' % (data), 'r') as f:
                w = f.readlines()
            weights = [int(i) for i in w] # Weight for edges
            print("weights:",len(weights))

        with open('/home/znyu/projects/Mole-GCN/src/data/%s/graph_labels.txt' % (data), 'r') as f:
            l = f.readlines()
        graph_labels = [int(i) for i in l]  # labels for all graphs
        print("labels:", len(graph_labels))

        with open('/home/znyu/projects/Mole-GCN/src/data/%s/node_labels.txt' % (data), 'r') as f:
            nl = f.readlines()
        node_labels = [int(i[-2]) for i in nl]  # labels for all graphs
        print("nodes_labels", len(node_labels))

        G_edges = []  # Edges for all graphs
        G_weight = []

        if self.args.edge_weight:
            for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
                edge = []  # Edges for one graph
                for e in range(len(edges)):
                    if g[edges[e][0] - 1] == i + 1:
                        edge.append(edges[e])

                    elif g[edges[e][0] - 1] == i + 2:
                        break
                G_edges.append(edge)
            G_weight = []
            for i in tqdm(range(len(graph_labels)), desc="Create weights", unit='graphs'):
                weight = []  # weights for edges in a graph
                for w in range(len(weights)):
                    if g[edges[w][0]-1] == i+1:
                        weight.append(weights[w])
                    elif g[edges[w][0]-1] == i + 2:
                        break
                G_weight.append(weight)
        else:
            for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
                edge = []  # Edges for one graph
                weight = []
                for e in range(len(edges)):
                    if g[edges[e][0] - 1] == i + 1:
                        edge.append(edges[e])
                        weight.append(1)
                    elif g[edges[e][0] - 1] == i + 2:
                        break
                G_edges.append(edge)
                G_weight.append(weight)

        g_list = []
        for i in tqdm(range(len(G_edges)), desc="Create original graph", unit='graphs'):
            g_list.append(self.gen_graph(G_edges[i], G_weight[i]))

        return G_data(g_list, node_labels, graph_labels)

    # Generate graph from original dataset
    def gen_graph(self, data, weights):
        edges = data
        weights = weights
        g1 = []
        for i in range(len(edges)):
            l = list(edges[i])
            l.append(weights[i])
            g1.append(tuple(l))

        g = nx.Graph()
        g.add_weighted_edges_from(g1)
        return g

