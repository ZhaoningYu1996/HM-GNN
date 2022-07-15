import networkx as nx
from tqdm import tqdm
import math
import numpy as np
import copy
import matplotlib.pyplot as plt

def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    print("degree:", degree)
    return degree.dot(adj).dot(degree)

class Generator(object):
    def __init__(self, data):
        self.data = data
        self.num_cliques = 0
        self.vocab = {}
        self.weight_vocab = {}
        self.count_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        self.whole_node_count = {}
        self.g_list = data.g_list
        self.node_labels = self.data.node_labels

    def gen_components(self, g_list):
        nodes_labels = self.node_labels
        G = nx.Graph()
        for g in tqdm(range(len(g_list)), desc='Gen components', unit='graph'):
            edge_list = []
            mcb = self.find_minimum_cycle_basis(g_list[g])

            mcb_tuple = [tuple(ele) for ele in mcb]

            edges = []
            for e in g_list[g].edges:
                count = 0
                for c in mcb:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            edges = list(set(edges))

            for e in edges:
                weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                edge = ((nodes_labels[e[0]-1], nodes_labels[e[1]-1]), weight)
                clique_id = self.add_to_vocab(edge)
                edge_list.append(clique_id)

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g_list[g]))
                ring = []
                for i in range(len(m)):
                    ring.append(nodes_labels[m[i]-1])
                cycle = (tuple(ring), weight)
                edge_list.append(self.add_to_vocab(cycle))

            check_point = {}

            for e in edge_list:
                self.add_weight(e, g)
                check_point[e] = 0
                if e not in self.whole_node_count:
                    self.whole_node_count[e] = 1
                else:
                    self.whole_node_count[e] += 1

            for e in edge_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    if check_point[e] == 1:
                        continue
                    else:
                        self.node_count[e] += 1
                check_point[e] = 1

            if len(edge_list) == 0:
                print("Something went wrong in graph{}.".format(g))
                new_pos = nx.spring_layout(g_list[g])
                nx.draw_networkx(g_list[g], new_pos, node_size=30, edge_color='black', font_size=3, font_color='purple')
                plt.show()
                G.add_edge("M{}".format(g + 1), "C{}".format(1),
                           weight=1)

            if g == 11157:
                print("number of edge list:", len(edge_list))
                print("graph 11158 edge list:", edge_list)

            for e in list(set(edge_list)):
                # G.add_edge("M{}".format(g + 1), "C{}".format(e),
                #            weight=(self.weight_vocab[(g + 1, e)] / len(mcb_tuple)))
                G.add_edge("M{}".format(g+1), "C{}".format(e), weight=(self.weight_vocab[(g+1, e)]/(len(edges)+len(mcb_tuple))))
                # G.add_edge("M{}".format(g + 1), "C{}".format(e), weight=1)

            for e in range(len(edges)):
                for i in range(e, len(edges)):
                    for j in edges[e]:
                        if j in edges[i]:
                            weight = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge = ((nodes_labels[edges[e][0] - 1], nodes_labels[edges[e][1] - 1]), weight)
                            weight_i = g_list[g].get_edge_data(edges[i][0], edges[i][1])['weight']
                            edge_i = ((nodes_labels[edges[i][0] - 1], nodes_labels[edges[i][1] - 1]), weight_i)
                            final_edge = tuple(sorted((self.add_to_vocab(edge), self.add_to_vocab(edge_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

            # print("check point 6")
            for m in range(len(mcb_tuple)):
                for i in range(m, len(mcb_tuple)):
                    for j in mcb_tuple[m]:
                        if j in mcb_tuple[i]:
                            weight = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring = []
                            for t in range(len(mcb_tuple[m])):
                                ring.append(nodes_labels[mcb_tuple[m][t] - 1])
                            cycle = (tuple(ring), weight)

                            weight_i = tuple(self.find_ring_weights(mcb_tuple[i], g_list[g]))
                            ring_i = []
                            for t in range(len(mcb_tuple[i])):
                                ring_i.append(nodes_labels[mcb_tuple[i][t] - 1])
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
                            edge_e = ((nodes_labels[edges[e][0] - 1], nodes_labels[edges[e][1] - 1]), weight_e)
                            weight_m = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring_m = []
                            for t in range(len(mcb_tuple[m])):
                                ring_m.append(nodes_labels[mcb_tuple[m][t] - 1])
                            cycle_m = (tuple(ring_m), weight_m)

                            final_edge = tuple(sorted((self.add_to_vocab(edge_e), self.add_to_vocab(cycle_m))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

        return G

    def add_weight(self, node_id, g):
        if (g+1, node_id) not in self.weight_vocab:
            self.weight_vocab[(g+1, node_id)] = 1
        else:
            self.weight_vocab[(g+1, node_id)] += 1

    def update_weight(self, g):
        for (u, v) in g.edges():
            if u[0] == 'M':
                g[u][v]['weight'] = g[u][v]['weight'] * (math.log(len(self.g_list) / (self.node_count[int(v[1:])])))
            else:
                g[u][v]['weight'] = g[u][v]['weight'] * (math.log(len(self.g_list) / (self.node_count[int(u[1:])])))
        return g

    def add_edge(self, g):
        edges = list(self.edge_count.keys())
        for i in edges:
            g.add_edge("C{}".format(i[0]), "C{}".format(i[1]), weight=math.exp(math.log(self.edge_count[i]/math.sqrt(self.whole_node_count[i[0]]*self.whole_node_count[i[1]]))))
        return g

    def add_to_vocab(self, clique):
        c = copy.deepcopy(clique[0])
        weight = copy.deepcopy(clique[1])
        for i in range(len(c)):
            if (c, weight) in self.vocab:
                return self.vocab[(c, weight)]
            else:
                c = self.shift_right(c)
                weight = self.shift_right(weight)
        self.vocab[(c, weight)] = len(self.vocab) + 1
        return self.vocab[(c, weight)]

    def update_vocab(self, g):
        keys = list(self.count_vocab.keys())
        count_list = []
        for i in keys:
            list_v = set(self.count_vocab[i])
            if len(list_v) > len(self.data.g_list)/2:
                count_list.append(self.vocab[i])
                g.remove_node('C{}'.format(self.vocab[i]))
                del self.vocab[i]
                del self.count_vocab[i]

        return g, sorted(count_list)

    def update_graph(self, g, count_list, l):
        mapping = {}
        print("length", len(count_list))
        if len(count_list) == 0:
            return g
        for i in range(len(count_list)):
            if i != 0:
                for j in range(count_list[i-1]+1, count_list[i]):
                    mapping['C{}'.format(j)] = 'C{}'.format(j-i)
        for i in range(count_list[-1]+1, l+1):
            mapping['C{}'.format(i)] = 'C{}'.format(i-len(count_list))
        updated_g = nx.relabel_nodes(g, mapping)
        return updated_g

    def gen_large_graph(self):
        g_list = self.data.g_list
        g = self.gen_components(g_list)
        g = self.update_weight(g)
        g_final = self.add_edge(g)
        return g_final, self.vocab

    def gen_cliques_vocab(self, g_list, nodes_labels):
        mcb_vocab = []
        edge_vocab = []
        vocab = {}
        for g in tqdm(range(len(g_list)), desc="Gen vocab", unit='graph'):
            mcb = self.find_minimum_cycle_basis(g_list[g])
            mcb_tuple = [tuple(ele) for ele in mcb]
            edges = []

            for e in g_list[g].edges:
                count = 0
                for c in mcb:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            edges = list(set(edges))
            # print("edges:", edges)

            for e in edges:
                weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                edge_vocab.append((((nodes_labels[e[0]-1], nodes_labels[e[1]-1]), weight), ((nodes_labels[e[1]-1], nodes_labels[e[0]-1]), weight)))
            if len(mcb_tuple) != 0:
                for m in mcb_tuple:
                    ring_list = []
                    # m = list(m)
                    # print(f"m: {m}")
                    weight = self.find_ring_weights(m, g_list[g])
                    # print(f"weight:{weight}")
                    for i in range(len(m)):
                        # print(f"sub m{m}")
                        ring = list(copy.deepcopy(m))
                        for j in range(len(m)):
                            ring[j] = nodes_labels[m[j]-1]
                        # print(f"ring {ring}")
                        ring_list.append((tuple(ring), tuple(weight)))
                        m = tuple(self.shift_right(list(m)))
                        weight = self.shift_right(weight)
                    ring_tuple = tuple(ring_list)
                    mcb_vocab.append(ring_tuple)
        # print(f"mcb_vocab{mcb_vocab}")
        mcb_vocab = list(set(map(tuple, mcb_vocab)))
        edge_vocab = list(set(tuple(edge_vocab)))
        # print(f"number of mcb and edge:{len(edge_vocab)}, {len(mcb_vocab)}")
        edge_vocab.extend(mcb_vocab)
        for i in range(len(edge_vocab)):
            vocab[edge_vocab[i]] = i+1

        # Sort G_vocab
        sorted_g_vocab = {}
        sorted_key = sorted(vocab, key=vocab.get)
        for w in sorted_key:
            sorted_g_vocab[w] = vocab[w]
        return sorted_g_vocab


    def gen_cliques(self, g_list):
        cliques = []
        for g in tqdm(range(len(g_list)), desc="Gen_cliques", unit="graphs"):
            cliques.append([])
            # Find mcb
            mcb = self.find_minimum_cycle_basis(g_list[g])
            # convert elements of mcb to tuple
            mcb_tuple = [tuple(ele) for ele in mcb]
            # print(mcb_tuple, g)

            # Find all edges not in cycles and add into cliques
            edges = []
            for e in g_list[g].edges:
                count = 0
                for c in mcb:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            cliques[g].extend(list(set(edges)))
            cliques[g].extend(mcb_tuple)
        return cliques

    def gen_vocab(self, cliques, nodes_labels):
        G_nodes = {}
        G_nodes_weights = {}
        G_vocab = {}
        for g in tqdm(range(len(cliques)), desc="Gen_vocab", unit="graphs"):
            # Generate G_nodes dictionary as vocabulary key tuple: (nodes, sum_of_edge_weights); value: 1 to number of cliques eg. ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)): 5
            # Using nodes label instead of nodes number
            for c in cliques[g]:
                le = []  # Nodes label lists
                lw = []  # Weights list of edges
                l_all = []
                if len(c) == 2:  # Bonds
                    for i in c:
                        le.append(nodes_labels[i - 1])
                    dic = self.data.g_list[g].get_edge_data(c[0], c[1])
                    lw.append(dic['weight'])
                else:  # Rings
                    for i in c:
                        le.append(nodes_labels[i - 1])
                    for i in range(len(c) - 1):
                        dic = self.data.g_list[g].get_edge_data(c[i], c[i + 1])
                        lw.append(dic['weight'])
                    dic = self.data.g_list[g].get_edge_data(c[-1], c[0])
                    lw.append(dic['weight'])
                global_cliques = tuple(le)
                l_all.append(global_cliques)
                if len(lw) == 1:
                    l_all.append(lw[0])
                else:
                    l_all.append(tuple(lw))
                l_all = tuple(l_all)
                if global_cliques not in G_nodes:
                    G_nodes[global_cliques] = len(G_nodes) + 1  # Value of G_nodes start from 1
                if l_all not in G_nodes_weights:
                    G_nodes_weights[l_all] = len(G_nodes_weights) + 1
        # Process G_nodes_weights
        keys = list(G_nodes_weights.keys())
        # Find the duplicated rings and bonds
        count_list = []  # values of all duplicated rings and bonds in dictionary
        for i in tqdm(range(len(keys)), desc="Find duplicated", unit="graph"):
            i_list = list(keys[i][0])
            for j in range(i + 1, len(keys)):
                inner_list = []
                if len(keys[i][0]) == 2:
                    inner_list.append(i_list)
                    inner_list.append(self.shift_right(i_list))
                    if list(keys[j][0]) in inner_list:
                        if keys[j][1] == keys[i][1]:
                            count_list.append(j)
                elif len(keys[i][0]) > 2:
                    i_inner_list = i_list
                    i_w_list = list(keys[i][1])
                    for t in range(len(i_list)):
                        if i_inner_list == list(keys[j][0]) and i_w_list == list(keys[j][1]):
                            count_list.append(j)
                        else:
                            i_inner_list = self.shift_right(i_inner_list)
                            i_w_list = self.shift_right(i_w_list)

        # Delete all duplicated rings and bonds
        g_nodes_weights = {key: val for key, val in G_nodes_weights.items() if val - 1 not in count_list}

        # Revalue the dictionary
        i = 1
        for key in g_nodes_weights.keys():
            G_vocab[key] = i
            i += 1

        # Extend G_vocab with all possible cliques sequences
        keys_list = list(G_vocab.keys())
        for v in tqdm(keys_list, desc="Extend vocab", unit='graphs'):
            lv = []
            lvw = []
            if len(v[0]) == 2:
                lv.append(v)
                if (tuple(self.shift_right(list(v[0]))), v[1]) not in lv:
                    lv.append((tuple(self.shift_right(list(v[0]))), v[1]))
                lv = tuple(lv)
                G_vocab[lv] = G_vocab[v]
                del G_vocab[v]
            elif len(v[0]) > 2:
                ring_list = []
                rings = v[0]
                weights = v[1]
                for i in range(len(v[0])):
                    lv.append(rings)
                    lvw.append(weights)
                    rings = tuple(self.shift_right(list(rings)))
                    weights = tuple(self.shift_right(list(weights)))
                for i in range(len(v[0])):
                    if (lv[i], lvw[i]) not in ring_list:
                        ring_list.append((lv[i], lvw[i]))
                ring_list = tuple(ring_list)
                G_vocab[ring_list] = G_vocab[v]
                del G_vocab[v]

        # Sort G_vocab
        sorted_g_vocab = {}
        sorted_key = sorted(G_vocab, key=G_vocab.get)
        for w in sorted_key:
            sorted_g_vocab[w] = G_vocab[w]
        return sorted_g_vocab

    def gen_edges(self, cliques, sorted_G_vocab, nodes_labels):
        # Count edges between cliques
        count_edges = []
        count_nodes = []
        all_nodes = []
        for g in tqdm(range(len(cliques)), desc="count node edge", unit="graph"):
            edges_dic = {}
            nodes_dic = {}
            clique_edges = []

            # Count all nodes
            for c in cliques[g]:
                x = list(c)
                index = self.find_index(x, self.data.g_list[g], sorted_G_vocab, nodes_labels)
                if index not in nodes_dic.keys():
                    nodes_dic[index] = 1
                else:
                    nodes_dic[index] += 1
                all_nodes.append(index)
            count_nodes.append(nodes_dic)
            # Find all edges between cliques
            for i in range(len(cliques[g])):
                for j in cliques[g][i]:
                    for t in range(i + 1, len(cliques[g])):
                        if j in cliques[g][t]:
                            clique_edges.append((cliques[g][i], cliques[g][t]))
            clique_edges = list(set(clique_edges))  # Delete duplicated edges

            # Count all edges
            for x, y in clique_edges:
                edge_keys = []
                index_x = self.find_index(x, self.data.g_list[g], sorted_G_vocab, nodes_labels)
                index_y = self.find_index(y, self.data.g_list[g], sorted_G_vocab, nodes_labels)
                edge_keys.append(index_x)
                edge_keys.append(index_y)
                edge_keys = sorted(edge_keys)
                if tuple(edge_keys) not in edges_dic.keys():
                    edges_dic[tuple(edge_keys)] = 1
                else:
                    edges_dic[tuple(edge_keys)] += 1
            count_edges.append(edges_dic)

        # Calculate edge weight
        edge_weight = {}
        for i in count_edges:
            for key in i.keys():
                if key not in edge_weight:
                    count = 0
                    for j in count_edges:
                        for keyj in j.keys():
                            if keyj == key:
                                count += 1
                    edge_weight[key] = count
        node_count = {}
        for i in count_nodes:
            for key in i.keys():
                if key not in node_count:
                    count = 0
                    for j in count_nodes:
                        for keyj in j.keys():
                            if key == keyj:
                                count += 1
                    node_count[key] = count
        cliques_weight = {}
        for i in count_edges:
            for key in i.keys():
                if key not in cliques_weight:
                    weight = math.log(
                        edge_weight[key] * len(count_edges) / (node_count[key[0]] * node_count[key[1]]))
                    if weight <= 0:
                        cliques_weight[key] = 0
                    else:
                        cliques_weight[key] = weight

        moles_weight = []
        mole_count = {}
        for i in count_nodes:
            for key in i.keys():
                if key not in mole_count:
                    mole_count[key] = 1
                else:
                    mole_count[key] += 1
        # print(mole_count)
        for i in count_nodes:
            m_weight = {}
            for key in i.keys():
                m_w = i[key] * math.log((len(count_nodes) + 1 / mole_count[key]) + 1)
                m_weight[key] = m_w
            moles_weight.append(m_weight)

        return count_edges, count_nodes, cliques_weight, moles_weight

    def gen_final_graph(self, count_edges, count_nodes, cliques_weight, moles_weight, sorted_G_vocab):
        # Generate the large graph
        c_nodes = []  # Cliques nodes
        for i in range(len(sorted_G_vocab)):
            c_nodes.append("C{}".format(i + 1))
        G_large = nx.Graph()
        G_large.add_nodes_from(c_nodes)
        m_nodes = []
        for i in range(len(self.data.g_list)):
            m_nodes.append("M{}".format(i + 1))
        G_large.add_nodes_from(m_nodes)

        # Add edges between cliques
        for i in count_edges:
            for key in i.keys():
                # If ignore the edge which weights are 0
                if cliques_weight[key] == 0:
                    continue
                else:
                    G_large.add_edge("C{}".format(key[0]), "C{}".format(key[1]), weight=cliques_weight[key])

        # Add edges between clique and mole
        for i in range(len(count_nodes)):
            for key in count_nodes[i].keys():
                G_large.add_edge("M{}".format(i + 1), "C{}".format(key), weight=moles_weight[i][key])
        return G_large

    def find_minimum_cycle_basis(self, g):
        return nx.cycle_basis(g)

    def shift_right(self, l):
        if type(l) == int:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return [l[-1]] + l[:-1]


    def find_ring_weights(self, ring, g):
        weight_list = []
        for i in range(len(ring)-1):
            weight = g.get_edge_data(ring[i], ring[i+1])['weight']
            weight_list.append(weight)
        weight = g.get_edge_data(ring[-1], ring[0])['weight']
        weight_list.append(weight)
        return weight_list


    def find_index(self, node, graph, vocab, nodes_labels):
        nodes_labels = nodes_labels
        index = -1
        x = list(node)
        g = graph
        sorted_G_vocab = vocab
        vocab_keys = list(sorted_G_vocab.keys())
        if len(x) == 2:
            w_inner = g.get_edge_data(x[0], x[1])['weight']
            for i in range(len(x)):
                x[i] = nodes_labels[x[i] - 1]
            for key in vocab_keys:
                if (tuple(x), w_inner) in key:
                    index = sorted_G_vocab[key]
        else:
            weight_list = []
            for i in range(len(x) - 1):
                e_label = g.get_edge_data(x[i], x[i + 1])
                weight_list.append(e_label['weight'])
            e_label = g.get_edge_data(x[-1], x[0])
            weight_list.append(e_label['weight'])
            for i in range(len(x)):
                x[i] = nodes_labels[x[i] - 1]
            for key in vocab_keys:
                if (tuple(x), tuple(weight_list)) in key:
                    index = sorted_G_vocab[key]
        return index