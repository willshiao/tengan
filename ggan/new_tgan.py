import torch
import torch.nn as nn
from opt_einsum import contract

class NewCPTensorGenerator(nn.Module):
    def __init__(self, latent_dim=100, layer_size=128, num_nodes=500, rank=30, extra_dim=False, num_views=10):
        super(NewCPTensorGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim
        self.num_views = num_views

        shared_layers = [
            nn.Linear(latent_dim, layer_size),
            nn.ReLU(inplace=True),
            # New block
            nn.Linear(layer_size, layer_size * 2),
            nn.BatchNorm1d(layer_size * 2),
            nn.ReLU(inplace=True),
            # New block
            nn.Linear(layer_size * 2, layer_size * 4),
            nn.BatchNorm1d(layer_size * 4),
            nn.ReLU(inplace=True)
        ]

        output_layers = [
            [
                nn.Linear(layer_size * 4, layer_size * 2),
                nn.ReLU(),
                nn.Linear(layer_size * 2, num_nodes * rank),
                nn.Sigmoid()
                # nn.BatchNorm1d(layer_size * 4),
            ] for _ in range(2)
        ]
        view_layer = [
            nn.Linear(layer_size * 4, layer_size * 2),
            nn.ReLU(),
            nn.Linear(layer_size * 2, num_views * rank),
            nn.Sigmoid()
            # nn.BatchNorm1d(layer_size * 4),
        ]

        self.shared = nn.Sequential(*shared_layers)
        self.output1 = nn.Sequential(*output_layers[0])
        self.output2 = nn.Sequential(*output_layers[1])
        self.output3 = nn.Sequential(*view_layer)
        self.output_factors = False
    
    def set_factor_output(self, new_val):
        self.output_factors = new_val
        return True

    def forward(self, noise):
        batch_sz = noise.shape[0]
        S = self.shared(noise)
        A = self.output1(S).view(batch_sz, self.num_nodes, self.rank)
        B = self.output2(S).view(batch_sz, self.num_nodes, self.rank)
        C = self.output3(S).view(batch_sz, self.num_views, self.rank)
        out = contract('faz,fbz,fcz->fabc', A, B, C, backend='torch')
        # print('out shape: ', out.shape)
        out = out.permute(0, 3, 1, 2)
        # print('out (permuted) shape: ', out.shape)

        if self.output_factors:
            return (out, (A, B, C))
        else:
            return out

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
