from networkx.classes.function import number_of_edges
from networkx.generators.random_graphs import barabasi_albert_graph
import numpy as np
from tqdm import tqdm
import argparse
import dgl
import dgl.data
import torch
from utils.gin import TwoGIN
from utils.ops import load_data
from sklearn.model_selection import KFold, StratifiedKFold
import random
import statistics as st
import pickle


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=14, help='seed')
    parser.add_argument('-data', default='MUTAG', help='data folder name')
    parser.add_argument('-num_epochs', type=int, default=2000, help='epochs')
    parser.add_argument('-batch_size', type=int, default=188, help='batch size')
    parser.add_argument('-lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('-w_d', type=float, default=0.0005, help='weight decay')
    parser.add_argument('-l_num', type=int, default=4, help='layer num')
    parser.add_argument('-h_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('-drop_n', type=float, default=0.2, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-device', type=int, default=0, help='device')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument(
        '-learn_eps', action="store_true",
        help='learn the epsilon weighting')
    parser.add_argument('-iters_per_epoch', type=int, default=50, help='iterations per epoch')
    parser.add_argument('-num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
    parser.add_argument('--lr_schedule_patience', type=int, default=20)
    parser.add_argument('--min_lr', type=float, default=0.001)
    args, _ = parser.parse_known_args()
    return args

def sep_data(labels, seed):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    test_idx = []
    train_idx = []
    for train_val_index, test_index in skf.split(np.zeros(len(labels)), labels):
        test_index = list(test_index)
        train_val_index = list(train_val_index)
        test_idx.append(test_index)
        train_idx.append(train_val_index)
    return train_idx, test_idx

def gen_features_labels(labels, g, num_cliques):
    emtpy_labels = torch.zeros(num_cliques, dtype=torch.long)
    labels = torch.cat((labels, emtpy_labels), 0)
    features_c = torch.eye(num_cliques)
    num_moles = g.num_nodes() - num_cliques
    features_m = torch.zeros(num_moles, num_cliques)
    for i in range(len(g.edges()[0])):
        if g.edges()[0][i] < num_moles:
            features_m[g.edges()[0][i]][g.edges()[1][i] - num_moles] = 1
        else:
            if g.edges()[1][i] < num_moles:
                features_m[g.edges()[1][i]][g.edges()[0][i] - num_moles] = 1
    features = torch.cat((features_m, features_c), 0)
    return features, labels

def accuracy(outputs, y):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == y).item() / len(preds)

def count_correct(outputs, y):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == y).item()

def average(lst):
    return sum(lst) / len(lst)

def load_subtensor(nfeat, labels, edge_weight, EID, seeds, input_nodes, device):
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    batch_edge_weight = []
    for i in EID:
        batch_edge_weight.append(edge_weight[i])
    return batch_inputs, batch_labels, batch_edge_weight

def train_and_evaluate(args, model, num_cliques, loss_fcn, feat, labels, graphs, dataloader, val_dataloader, edge_weight, optimizer, device):
    model.train()  
    count_train = 0
    length_train = 0
    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        selected_idx = seeds
        IDs = []
        for block in blocks:
            IDs.append(block.edata[dgl.EID])
        batch_graph = []
        for i in selected_idx:
            if i < len(graphs):
                batch_graph.append(graphs[i])
        batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, labels, edge_weight, IDs,
                                                        seeds, input_nodes, device)
        blocks = [block.int().to(device) for block in blocks]
        batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph, num_cliques)
        loss = loss_fcn(batch_pred.to(device), batch_labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count_train += count_correct(batch_pred, batch_labels)
        length_train += len(batch_labels)
    train_acc = count_train / length_train
    model.eval()
    with torch.no_grad():
        count = 0
        length = 0
        
        for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
            selected_idx = seeds
            IDs = []
            for block in blocks:
                IDs.append(block.edata[dgl.EID])
            batch_graph = []
            for i in selected_idx:
                if i < len(graphs):
                    batch_graph.append(graphs[i])
            batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, labels, edge_weight, IDs,
                                                            seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph, num_cliques)
            length += len(batch_labels)
            count += count_correct(batch_pred, batch_labels)
        val_acc = count / length
        return train_acc, val_acc
        


def evaluate(model, graph, features, labels, train_mask, mask, norm_edge_weight, graphs, num_cliques, device):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features, norm_edge_weight, graphs, num_cliques)
        train_logits = logits[train_mask].to(device)
        train_labels = labels[train_mask].to(device)
        logits = logits[mask].to(device)
        labels = labels[mask].to(device)
        _, indices_t = torch.max(train_logits, dim=1)
        correct_t = torch.sum(indices_t == train_labels)
        train_acc = correct_t.item() * 1.0 / len(train_labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return train_acc, correct.item() * 1.0 / len(labels)

def evaluate_dgl(model, graph, features, labels, train_mask, val_mask, edge_weight, dataloader, num_cliques, device):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features, edge_weight, dataloader, num_cliques)
        train_logits = logits[train_mask].to(device)
        train_labels = labels[train_mask].to(device)
        logits = logits[val_mask].to(device)
        labels = labels[val_mask].to(device)
        _, indices_t = torch.max(train_logits, dim=1)
        correct_t = torch.sum(indices_t == train_labels)
        train_acc = correct_t.item() * 1.0 / len(train_labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return train_acc, correct.item() * 1.0 / len(labels)

def main():
    CUDA_LAUNCH_BLOCKING=1
    args = get_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.data == 'PROTEINS':
        number_of_graphs = 1113
    elif args.data == 'PTC_MR':
        number_of_graphs = 344
    elif args.data == 'MUTAG':
        number_of_graphs = 188
    elif args.data == 'NCI1':
        number_of_graphs = 4110
    elif args.data == 'Mutagenicity':
        number_of_graphs = 4337
    with open('preprocessed_datasets/' + args.data, 'rb') as input_file:
        g = pickle.load(input_file)
    num_cliques = int(g.number_of_nodes()) - number_of_graphs
    # print(num_cliques)
    labels = g.ndata['labels']
    features = g.ndata['feat']
    in_feats = features.size()[1]

    edge_weight = g.edata['edge_weight'].to(device)

    g = g.to(device)
    node_features = features.to(device)
    labels.to(device)

    graphs, num_classes = load_data(args.data, args.degree_as_tag)
    print(len(graphs))
    
    max_val = 0
    best_seed = 14
    for step in range(50):
        seed=109
        val_acc = []
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        train_idx, valid_idx = sep_data(labels[:number_of_graphs], seed)
        print(valid_idx)
        for i in range(10):
            train_index = train_idx[i]
            valid_index = valid_idx[i]
            train_mask = [True if x in train_index else False for x in range(int(g.num_nodes()))]
            train_mask = np.array(train_mask)
            valid_mask = [True if x in valid_index else False for x in range(int(g.num_nodes()))]
            valid_mask = np.array(valid_mask)
            g.ndata['train_mask'] = torch.from_numpy(train_mask).to(device)
            g.ndata['val_mask'] = torch.from_numpy(valid_mask).to(device)
            train_mask = g.ndata['train_mask'].to(device)
            valid_mask = g.ndata['val_mask'].to(device)

            train_nid = torch.nonzero(train_mask, as_tuple=True)[0].to(device)
            val_nid = torch.nonzero(valid_mask, as_tuple=True)[0].to(device)
            g = g.to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                train_nid,
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
            val_dataloader = dgl.dataloading.NodeDataLoader(
                g,
                val_nid,
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )

            gin = TwoGIN(args.l_num, 2, in_feats, graphs[0].node_features.shape[1], args.h_dim, 2, args.drop_n, args.drop_c, args.learn_eps, 'sum', 'sum').to(device)
            loss_fcn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(gin.parameters(), lr=args.lr, weight_decay=args.w_d)
            acc_mx = 0.0

            for epoch in tqdm(range(args.num_epochs), desc='epochs', unit='epoch'):
                train_acc, acc = train_and_evaluate(args, gin, num_cliques, loss_fcn, node_features, labels, graphs, dataloader, val_dataloader, edge_weight, optimizer, device)
                final_acc = train_acc
                if acc > acc_mx:
                    acc_mx = acc
                    train_mx = train_acc
                if optimizer.param_groups[0]['lr'] < args.min_lr:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
            val_acc.append(acc_mx)
            print(f'The best val accuracy of model with {i} fold is {acc_mx}')
        if average(val_acc) > max_val:
            max_val = average(val_acc)
            best_seed = seed
        print(f"The total val accuracy of model with seed {seed} is {average(val_acc)}")
        print(f"Standard Deviation of model is {st.stdev(val_acc)}")
        f = open(f"result_{args.data}.txt", "a")
        f.write(f'Batch size: {args.batch_size}. Learning rate: {args.lr}. Hidden dim: {args.h_dim}. Dropout net: {args.drop_n}. Dropout graph: {args.drop_c} \n')
        f.write(f"The Best val accuracy is {average(val_acc)}, the std is {st.stdev(val_acc)}, the seed is {seed}.  \n")
        f.close()
    print(f'Best val acc is {max_val}, best seed is {best_seed}.')

if __name__ == '__main__':
    main()
