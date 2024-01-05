"""
GNN models

"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F

class GIN(torch.nn.Module):

    """GIN for graph classification"""

    def __init__(self, dim_h, dim_in):
        super(GIN, self).__init__()
							  )
        self.conv1 = GINConv(
            Sequential(Linear(dim_in, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))


        self.lin1 = Linear(dim_h*2, dim_h*12)
        self.lin2 = Linear(dim_h*12, 2)

        self.b1 = BatchNorm1d(dim_h*12)
        self.b2 = BatchNorm1d(dim_h*6)

        self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self,
				x,
				edge_index,
				batch=None,
				):
        
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = self.b1(h).relu()
        h = self.dropout(h)
        h = self.lin2(h)

        # h = torch.squeeze(h)

        h = F.sigmoid(h)

        return h