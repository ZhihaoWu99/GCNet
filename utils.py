import torch
import configparser
from texttable import Texttable
from sklearn import metrics
import torch.nn.functional as F
from args import parameter_parser


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def set_args():
    args = parameter_parser()
    args.device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    conf = configparser.ConfigParser()
    config_path = './config' + '.ini'
    conf.read(config_path)
    args.lr = conf.getfloat(args.dataset, 'lr')
    args.wd = conf.getfloat(args.dataset, 'wd')
    args.alpha = conf.getfloat(args.dataset, 'alpha')
    args.beta = conf.getfloat(args.dataset, 'beta')
    args.num_epoch = conf.getint(args.dataset, 'num_epoch')
    args.num_pse = conf.getint(args.dataset, 'num_pse')
    assert args.model in ['GCN', 'GCNet']
    if args.model == 'GCN':
        args.basic = True
    else:
        args.basic = False
    # Set the hidden dim to 32 as UAI has 19 classes
    if args.dataset == 'UAI':
        args.hdim = 32
    return args


def get_evaluation_results(labels_true, labels_pred, idx_train, idx_val, idx_test):
    acc, f1 = {}, {}
    acc['tr'] = metrics.accuracy_score(labels_true[idx_train], labels_pred[idx_train]) * 100
    f1['tr'] = metrics.f1_score(labels_true[idx_train], labels_pred[idx_train], average='macro') * 100
    acc['val'] = metrics.accuracy_score(labels_true[idx_val], labels_pred[idx_val]) * 100
    f1['val'] = metrics.f1_score(labels_true[idx_val], labels_pred[idx_val], average='macro') * 100
    acc['test'] = metrics.accuracy_score(labels_true[idx_test], labels_pred[idx_test]) * 100
    f1['test'] = metrics.f1_score(labels_true[idx_test], labels_pred[idx_test], average='macro') * 100
    return acc, f1


class Criterion:
    def __init__(self, N, idx_train, idx_unlabeled, labels, device):
        self.N = N
        self.labels = labels
        self.idx_unlabeled = idx_unlabeled
        self.idx_train = idx_train
        self.device = device
        self.s_mask_pos = None
        self.s_mask_neg = None
        self.init_mask()

    def init_mask(self):
        l_mask = torch.zeros(self.N, self.N).bool().to(self.device)
        for idx in self.idx_train:
            l_mask[idx, self.idx_train] = True
        s_labels = self.labels[:, None]
        s_mask = l_mask * torch.eq(s_labels, s_labels.t()).bool().to(self.device)
        eye = torch.eye(s_mask.shape[0]).bool().to(self.device)
        self.s_mask_pos = s_mask.masked_fill(eye, 0).float()
        self.s_mask_neg = (~s_mask).float()

    def pseudo_labels(self, num_labels, embedding):
        max_value, _ = torch.max(embedding, dim=1)
        _, idx = torch.sort(max_value, descending=True)
        idx = idx.cpu().numpy()
        idx_trust = [i for i in idx if i in set(self.idx_unlabeled)]
        return idx_trust[:num_labels]

    def compute_loss(self, args, embedding, output):
        if not args.basic:
            s_z = F.softmax(embedding, dim=1)
            s_z = F.normalize(s_z, p=2, dim=1)
            dot_s_z = torch.mm(s_z, s_z.t())
            s_pos_pairs_mean = (self.s_mask_pos * dot_s_z).sum() / (self.s_mask_pos.sum() + 1e-6)
            s_neg_pairs_mean = (self.s_mask_neg * dot_s_z).sum() / (self.s_mask_neg.sum() + 1e-6)
            loss1 = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss2 = 0.5 * s_neg_pairs_mean + (1 - s_pos_pairs_mean)
            loss = loss1 + loss2
        else:
            loss = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        return loss

    def update_masks(self, args, embedding, pred_labels):
        p_labels_ind = self.pseudo_labels(args.num_pse, F.softmax(embedding.detach(), dim=1))
        idx_label = self.idx_train + p_labels_ind
        l_mask = torch.zeros(self.N, self.N).bool().to(self.device)
        for idx in idx_label:
            l_mask[idx, idx_label] = True
        p_labels_ = torch.from_numpy(pred_labels).long()
        for idx in self.idx_train:
            p_labels_[idx] = self.labels[idx]
        p_labels = p_labels_[:, None]
        p_mask = l_mask * torch.eq(p_labels, p_labels.t()).bool().to(self.device)
        eye = torch.eye(p_mask.shape[0]).bool().to(self.device)
        self.s_mask_pos = p_mask.masked_fill(eye, 0).float()
        self.s_mask_neg = (~p_mask).float()
