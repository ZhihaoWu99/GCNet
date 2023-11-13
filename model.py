import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias, p=2, batch_norm=False, atl=False):
        super(GCNConv, self).__init__(aggr='add')
        self.p = p
        self.batch_norm = batch_norm
        self.atl = atl
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        if self.batch_norm:
            self.bn = BatchNorm1d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.batch_norm:
            self.bn.reset_parameters()

    def forward(self, x, edge_index_org, edge_index_knn=None, alpha=None):
        edge_index_org, _ = add_self_loops(edge_index_org, num_nodes=x.size(0))

        if self.atl:
            assert edge_index_knn is not None and alpha is not None
            edge_index_knn, _ = remove_self_loops(edge_index_knn)

            # Compute l_p norm
            row_knn, col_knn = edge_index_knn
            dist_knn = torch.pow(torch.norm(x[row_knn] - x[col_knn], dim=1).clamp(min=1e-12), self.p-2)

            # Compute the A_f
            edge_index_combined = torch.cat([edge_index_org, edge_index_knn], dim=1)
            weights_combined = torch.cat([torch.ones(edge_index_org.size(1)).to(edge_index_org.device), alpha * dist_knn])
            row, col = edge_index_combined
            deg = self.weighted_degree(edge_index_combined, weights_combined, x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * weights_combined
        else:
            edge_index_combined = edge_index_org
            row, col = edge_index_combined
            deg = degree(row, x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transformation.
        x = self.lin(x)

        # Propagation
        if self.batch_norm:
            return self.bn(self.propagate(edge_index_combined, x=x, norm=norm))
        else:
            return self.propagate(edge_index_combined, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def weighted_degree(self, edge_index, edge_weight, num_nodes):
        out_degree = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_index.device)
        out_degree.scatter_add_(0, edge_index[1], edge_weight)
        return out_degree


class GCN(torch.nn.Module):
    def __init__(self, hidden_dims, dropout, bias, p=2, atl=True):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        batch_norm = True if len(hidden_dims) > 5 else False
        self.convs.append(GCNConv(hidden_dims[0], hidden_dims[1], bias, p, batch_norm))
        for i in range(1, len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i+1], bias, p, batch_norm, atl))

    def forward(self, x, edge_index_org, edge_index_knn=None, alpha=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index_org, edge_index_knn, alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index_org, edge_index_knn, alpha)
        return x
