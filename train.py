import os
import warnings
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from args import parameter_parser
from utils import tab_printer, get_evaluation_results, Criterion
from Dataloader import load_data
from model import GCN
import matplotlib.pyplot as plt


def train(args):
    feature, org_edge_index, knn_edge_index, hidden_dims, labels, idx_train, idx_test, idx_val, idx_unlabeled = load_data(args)
    plt.axis('off')
    N = feature.shape[0]
    criterion = Criterion(N, idx_train, idx_unlabeled, labels, args.device)
    model = GCN(hidden_dims, args.dropout, args.bias, args.p, not args.basic).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    begin_time = time.time()
    best_emb = torch.Tensor()
    best_val_acc = 0.
    best_val_f1 = 0.

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            embedding = model(feature, org_edge_index, knn_edge_index, args.alpha)
            output = F.log_softmax(embedding, dim=1)
            if epoch % 10 == 0 and epoch > 0:
                criterion.update_masks(args, embedding, pred_labels)
            loss = criterion.compute_loss(args, embedding, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                embedding = model(feature, org_edge_index, knn_edge_index, args.alpha)
                pred_labels = torch.argmax(embedding, 1).cpu().detach().numpy()
                acc, f1 = get_evaluation_results(labels.cpu().numpy(), pred_labels, idx_train, idx_val, idx_test)
                if (acc['val'] > best_val_acc) and (f1['val'] > best_val_f1):
                    best_epoch, best_emb, best_val_acc, best_val_f1 = epoch, embedding, acc['val'], f1['val']
                pbar.set_postfix({'Best_itr': '{:}'.format(best_epoch),
                                  'Loss': '{:.4f}'.format(loss.item()),
                                  'ACC': '{:.2f}'.format(acc['test']),
                                  'F1': '{:.2f}'.format(f1['test'])})
                pbar.update(1)

    cost_time = time.time() - begin_time
    print("Best result epoch:{}".format(best_epoch))
    pred_labels = torch.argmax(best_emb, 1).cpu().detach().numpy()
    acc, f1 = get_evaluation_results(labels.cpu().numpy(), pred_labels, idx_train, idx_val, idx_test)

    return acc['test'], f1['test'], cost_time
