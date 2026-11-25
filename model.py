import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.known_class
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, self.nhid))

        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.nhid, self.nhid))

        self.cls = nn.Linear(self.nhid, self.num_classes+1)
        self.unknown_cls = nn.Linear(self.nhid, 1)


    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        return x

    def feat_classifier(self, x, edge_index, edge_weight=None):
        x = self.cls(x)
        return x
