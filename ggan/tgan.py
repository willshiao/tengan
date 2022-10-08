import torch
import torch.nn as nn
from ggan.layers import GraphConvolution
class NewLayeredMultiviewDiscriminator(nn.Module):
    def __init__(self, num_nodes=500, dropout=0.5, slices=6):
        super(NewLayeredMultiviewDiscriminator, self).__init__()
        self.num_nodes = num_nodes
        self.slices = slices

        self.gcns = nn.ModuleList([
            nn.ModuleList([
                GraphConvolution(self.num_nodes, self.num_nodes),
                nn.LeakyReLU(inplace=True),
                GraphConvolution(self.num_nodes, self.num_nodes),
                nn.LeakyReLU(inplace=True)
            ]) for __ in range(self.slices)
        ])

        self.final_layer = nn.Linear(self.num_nodes ** 2, 1)
        self.sig = nn.Sigmoid()

    def forward(self, ten):
        if torch.cuda.is_available():
            I = torch.eye(self.num_nodes).cuda()
        else:
            I = torch.eye(self.num_nodes)
        outs = []

        for k in range(self.slices):
            adj = ten[:, k, :, :]
            x = self.gcns[k][1](self.gcns[k][0](I, adj))
            x = self.gcns[k][3](self.gcns[k][2](I, x))
            # x = self.gcns[k](in_features=I, out_features=adj)
            # print(x.shape)
            # Maybe add a FC layer here?
            outs.append(x)

        stacked = torch.stack(outs, dim=1) \
            .view(ten.size(0), self.slices, self.num_nodes, self.num_nodes)
        pooled = nn.functional.max_pool3d(stacked, kernel_size=(self.slices, 1, 1))
        pooled = self.sig(pooled)
        x = pooled.view(pooled.size(0), -1)

        return self.final_layer(x), x
