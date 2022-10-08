import torch
import torch.nn as nn
from opt_einsum import contract

class RescalGenerator(nn.Module):
    def __init__(self, latent_dim=100, layer_size=128, num_nodes=500, rank=30, extra_dim=False, num_views=10):
        super(RescalGenerator, self).__init__()
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

        layer_A = [
            nn.Linear(layer_size * 4, layer_size * 2),
            nn.ReLU(),
            nn.Linear(layer_size * 2, num_nodes * rank),
            nn.Sigmoid()
            # nn.BatchNorm1d(layer_size * 4),
        ]
        tensor_layer = [
            nn.Linear(layer_size * 4, layer_size * 2),
            nn.ReLU(),
            nn.Linear(layer_size * 2, rank * rank * num_views),
            nn.Sigmoid()
            # nn.BatchNorm1d(layer_size * 4),
        ]

        self.shared = nn.Sequential(*shared_layers)
        self.out_a = nn.Sequential(*layer_A)
        self.out_r = nn.Sequential(*tensor_layer)
        self.output_factors = False
    
    def set_factor_output(self, new_val):
        self.output_factors = new_val
        return True

    def forward(self, noise):
        batch_sz = noise.shape[0]
        S = self.shared(noise)
        A = self.out_a(S).view(batch_sz, self.num_nodes, self.rank)
        R = self.out_r(S).view(batch_sz, self.rank, self.rank, self.num_views)
        # print('A: ', A.shape)
        # print('R: ', R.shape)
        A_t = torch.transpose(A, 1, 2)
        out = contract('fab,fbck,fcd->fadk', A, R, A_t, backend='torch')
        out = out.permute(0, 3, 1, 2)
        # left = torch.matmul(A, R)
        # out = torch.matmul(left, A_t)
        # print('out shape: ', out.shape)
        # print('out (permuted) shape: ', out.shape)

        if self.output_factors:
            return (out, (A, R))
        else:
            return out

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
