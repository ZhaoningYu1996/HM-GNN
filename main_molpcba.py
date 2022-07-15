
import argparse
from dgl.batch import batch
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.optim as optim
import dgl
from dgl.nn.pytorch import EdgeWeightNorm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from utils.model_ogb import TwoGraphGCN
import time
import pickle
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator, GraphPropPredDataset

cls_criterion = torch.nn.BCEWithLogitsLoss()

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.9,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('-drop_g', type=float, default=0.5, help='dropout ratio for first gin')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--first_hidden_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--hidden_dim', type=int, default=10,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--first_graph_hidden_dim', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=600,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molpcba",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-w_d', type=float, default=5e-5, help='weight decay')
    parser.add_argument(
        '-learn_eps', action="store_true",
        help='learn the epsilon weighting')
    parser.add_argument('-agg_method', type=str, default='mean', help='aggregation method')
    parser.add_argument('-fanouts', type=list, default=[100, 5, 5], help='output nodes number')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
    parser.add_argument('--lr_schedule_patience', type=int, default=20)
    parser.add_argument('--min_lr', type=float, default=0.0001)
    args, _ = parser.parse_known_args()
    return args

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

def train_and_evaluate(model, first_train, first_val, first_test, loss_fcn, feat, labels, train_dataloader, val_dataloader, test_dataloader, edge_weight, optimizer, evaluator, scheduler, device):
    model.train()
    bce_labels = labels
    train_preds = torch.tensor([], dtype=torch.float32).to(device)
    train_labels = torch.tensor([], dtype=torch.long).to(device)
    val_preds = torch.tensor([], dtype=torch.float32).to(device)
    val_labels = torch.tensor([], dtype=torch.long).to(device)
    test_preds = torch.tensor([], dtype=torch.float32).to(device)
    test_labels = torch.tensor([], dtype=torch.long).to(device)
    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
        selected_idx = seeds
        IDs = []
        for block in blocks:
            IDs.append(block.edata[dgl.EID])
        batch_graph = first_train[step].to(device)
        batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, bce_labels, edge_weight, IDs,
                                                        seeds, input_nodes, device)
        
        blocks = [block.int().to(device) for block in blocks]
        batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph).to(device)

        mask = batch_labels == batch_labels
        loss = loss_fcn(batch_pred[mask].to(torch.float32).to(device), batch_labels[mask].to(torch.float32).to(device))
        train_labels = torch.cat((train_labels, batch_labels), 0).to(device)
        train_preds = torch.cat((train_preds, batch_pred), 0).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds_eval = torch.tensor([], dtype=torch.float32).to(device)
        train_labels_eval = torch.tensor([], dtype=torch.long).to(device)
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            selected_idx = seeds
            IDs = []
            for block in blocks:
                IDs.append(block.edata[dgl.EID])
            batch_graph = first_train[step].to(device)
            batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, bce_labels, edge_weight, IDs,
                                                            seeds, input_nodes, device)
            
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph).to(device)
            train_preds_eval = torch.cat((train_preds, batch_pred), 0)
            train_labels_eval = torch.cat((train_labels, batch_labels), 0).to(device)

        input_dict = {"y_true": train_labels_eval, "y_pred": train_preds_eval}
        train_result_dict = evaluator.eval(input_dict)
        train_acc = train_result_dict['ap']
        for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
            selected_idx = seeds
            IDs = []
            for block in blocks:
                IDs.append(block.edata[dgl.EID])
            batch_graph = first_val[step].to(device)
            batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, bce_labels, edge_weight, IDs,
                                                            seeds, input_nodes, device)
            
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph).to(device)
            val_preds = torch.cat((val_preds, batch_pred), 0)
            val_labels = torch.cat((val_labels, batch_labels), 0).to(device)

        input_dict = {"y_true": val_labels, "y_pred": val_preds}
        val_result_dict = evaluator.eval(input_dict)
        val_acc = val_result_dict['ap']
        scheduler.step(-val_acc.item())

        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            selected_idx = seeds
            IDs = []
            for block in blocks:
                IDs.append(block.edata[dgl.EID])
            batch_graph = first_test[step].to(device)
            batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, bce_labels, edge_weight, IDs,
                                                            seeds, input_nodes, device)
            
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph).to(device)
            test_preds = torch.cat((test_preds, batch_pred), 0)
            test_labels = torch.cat((test_labels, batch_labels), 0).to(device)

        input_dict_test = {'y_true': test_labels, 'y_pred': test_preds}
        test_result_dict = evaluator.eval(input_dict_test)
        test_acc = test_result_dict['ap']
        print(f'Train ap is {train_acc}, Val ap is {val_acc}, Test ap is {test_acc}.')
        return val_acc, test_acc

def main():
    args = get_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device_cpu = torch.device("cpu")

    dataset = GraphPropPredDataset(name='ogbg-molpcba')
    labels = []
    for i in range(len(dataset)):
        g, l = dataset[i]
        labels.append(l)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dgl.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset = PygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    first_train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=False)
    first_val_loader = DataLoader(dataset[valid_idx], batch_size=args.batch_size, shuffle=False)
    first_test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)

    with open('ogb-molpcba', 'rb') as input_file:
        g = pickle.load(input_file)
    num_cliques = int(g.number_of_nodes()) - len(dataset)
    features = g.ndata['feat']
    in_feats = features.size()[1]

    edge_weight = g.edata['weight']
    norm = EdgeWeightNorm(norm='both')
    edge_weight = norm(g, edge_weight)
    edge_weight = edge_weight.to(device)

    mask_train = g.ndata['train_mask']
    mask_valid = g.ndata['val_mask']
    mask_test = g.ndata['test_mask']
    train_nid = torch.nonzero(mask_train, as_tuple=True)[0].to(device_cpu)
    val_nid = torch.nonzero(mask_valid, as_tuple=True)[0].to(device_cpu)
    test_nid = torch.nonzero(mask_test, as_tuple=True)[0].to(device_cpu)
    evaluator = Evaluator(name="ogbg-molpcba")

    features = features
    labels = np.array(labels,dtype=float)
    labels = torch.tensor(labels, dtype=torch.float)
    
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
                g.to('cpu'),
                train_nid,
                sampler,
                device=device_cpu,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
    val_dataloader = dgl.dataloading.NodeDataLoader(
                g.to('cpu'),
                val_nid,
                sampler,
                device=device_cpu,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
    test_dataloader = dgl.dataloading.NodeDataLoader(
                g.to('cpu'),
                test_nid,
                sampler,
                device=device_cpu,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
    
    model = TwoGraphGCN(args.num_layer, 2, in_feats, args.hidden_dim, args.first_hidden_dim, args.first_graph_hidden_dim, 128, args.drop_ratio, args.drop_g, args.learn_eps, args.agg_method, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=args.lr_reduce_factor,
                                                    patience=args.lr_schedule_patience,
                                                    verbose=True)
    val_mx = 0.0
    test_mx = 0.0
    first_train = []
    first_val = []
    first_test = []
    for step, batch in enumerate(first_train_loader):
        first_train.append(batch)
    for step, batch in enumerate(first_val_loader):
        first_val.append(batch)
    for step, batch in enumerate(first_test_loader):
        first_test.append(batch)
    for epoch in tqdm(range(args.num_epochs), desc='epochs', unit='epoch'):
        val_acc, test_acc = train_and_evaluate(model, first_train, first_val, first_test, cls_criterion, features, labels, train_dataloader, val_dataloader, test_dataloader, edge_weight, optimizer, evaluator, scheduler, device)
        if val_acc > val_mx:
            val_mx = val_acc
            test_mx = test_acc
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            print("\n!! LR EQUAL TO MIN LR SET.")
            break

    f = open(f"result_{args.dataset}.txt", "a")
    f.write(f'The batch size is {args.batch_size}. The first graph hidden dim is {args.first_graph_hidden_dim}. The hidden dim is {args.hidden_dim}. The learning rate is {args.lr}. The min learning rate is {args.min_lr} The dropout for first GIN is {args.drop_g}. The dropout for second GIN is {args.drop_ratio}. The number of epochs is {args.num_epochs}. The aggregation method is {args.agg_method}. The fanouts is {args.fanouts}. The leaning rate reduce factor is {args.lr_reduce_factor}. The learning rate scheduler patience is {args.lr_schedule_patience}. \n')
    f.write(f"The Best val roc auc is {val_mx}, the best test roc auc is {test_mx}. \n")
    f.close()


if __name__ == '__main__':
    main()
