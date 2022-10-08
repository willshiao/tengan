#!/usr/bin/env python
# coding: utf-8

# In[1]:

from genericpath import exists
#import graph_tool as gt
#from graph_tool.spectral import adjacency
import networkit as nk
from dsloader.tensor_creator import create_tensors
import wandb
from training import Trainer
from torch.utils.data.sampler import RandomSampler
from ggan import ModelZoo
from torch.utils.data import Dataset
import json
import numpy as np
import random
import torch
import torch.utils.data as utils
import sys
import os
from gutils.general import save_pickle, load_pickle

from os import path
from pathlib import Path


# In[2]:


random.seed(199231318)
np.random.seed(681212354)

cuda = True if torch.cuda.is_available() else False
print('CUDA is enabled' if cuda else 'CUDA is not enabled')

VERSION = '1.0.0'
ALLOW_OVERRIDE = False

opt = {
    'n_epochs': 5000,  # number of epochs of training
    'batch_size': 64,  # size of the batches
    'n_permutations': 5020,  # number of permutations of the graph
    'gen_lr': 0.001,  # adam: learning rate
    'disc_lr': 0.0001,  # learning rate for discriminator
    # (b1, b2): decay of first order momentum of gradient & first order momentum of gradient
    'betas': (0.5, 0.999),
    'n_cpu': 32,  # number of cpu threads to use during batch generation
    'latent_dim': 100,  # dimensionality of the latent space
    'rank': 100,  # rank used for bilinear size
    'gen_layer_size': 128,  # size of layers in generator
    'disc_layer_size': 1024,  # size of layers in discriminator
    'sample_interval': 800,  # interval between graph sampling
    'save_interval': 250,  # interval between model saving
    'print_interval': 100,  # interval between printing loss info
    'plot_interval': 200,  # interval between plotting
    'loss_interval': 10,  # interval between loss sampling
    'model_name': 'TenGAN',
    'slice_size': 50,  # graph slice size
    'fully_random': False,
    'comment': '',
    # generator class name, used by ModelZoo
    'gen_class': 'NewCPTensorGenerator',
    # discriminator class name, used by ModelZoo
    'disc_class': 'NewLayeredMultiviewDiscriminator',
    'dataset': 'football_small',
    'tensor_slices': 6,
    'rank_lambda': 0,
    'penalty_type': 'fro',
    'n_graph_sample_batches': 10,
    'rank_penalty_method': 'A', # A, B, or C
    'eval_every': 0,
    'sampling_method': 'random_walk',
    'eval_method': 'multiview',
    'critic_iterations': 1, # update generator every critic_iterations
    'generator_iterations': 2, # update discriminator every generator_iterations
    'version': VERSION,
    'uses_graphtool': False,
    'max_eval_rank': 40,  # max eval rank for tensor-based eval,
    'tensor_eval_samples': 50,  # number of samples for tensor-based eval
    'cache_data': True # whether or not to cache the dataset
}

for arg in sys.argv[1:]:
    pieces = arg.lstrip('-').split('=')
    if len(pieces) != 2:
        print(f'Invalid argument format: {arg}')
        sys.exit(1)
    l, r = pieces
    setting_ver = False

    # Get type of existing argument
    if l.lower() == 'version':
        VERSION = r
        setting_ver = True
        print(f'Setting version to {VERSION}')
    elif isinstance(opt[l], int):
        # convert to int
        opt[l] = int(r)
    elif isinstance(opt[l], str):
        opt[l] = r
    elif isinstance(opt[l], bool):
        lower_r = r.lower()
        if lower_r == 'true':
            opt[l] = True
        elif lower_r == 'false':
            opt[l] = False
        else:
            print(f'Invalid option for binary arg {l}')
            sys.exit(2)
    elif isinstance(opt[l], float):
        opt[l] = float(r)
    else:
        print('Unknown opt type for {l}')
        sys.exit(2)

    if not setting_ver:
        print(f'Setting opt[{l}] to {opt[l]}')
if 'gtools' in opt['dataset']:
    opt['uses_graphtool'] = True
    print('Setting uses_graphtool to True')

CACHE_PATH = 'cache'
BASE_PATH = 'results/'
PATH_PREFIX = path.join(BASE_PATH, '{}_v{}'.format(opt['model_name'], VERSION))
SAVE_PATH = path.join(PATH_PREFIX, 'models/')
IMG_PATH = path.join(PATH_PREFIX, 'images/')
STATS_PATH = path.join(PATH_PREFIX, 'stats/')


# In[7]:


if path.exists(PATH_PREFIX):
    if ALLOW_OVERRIDE:
        print(
            f'WARNING: ERROR: path ({PATH_PREFIX}) already exists, but ALLOW_OVERRIDE is set')
    else:
        raise Exception(
            f'ERROR: path ({PATH_PREFIX}) already exists, no files created')

to_make = [SAVE_PATH, IMG_PATH, STATS_PATH, CACHE_PATH]
for p in to_make:
    Path(p).mkdir(parents=True, exist_ok=True)

print('Saving params')
with open(path.join(PATH_PREFIX, 'params.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4, separators=(',', ': '))
wandb.init(config=opt, project='tensor-gan')

dataset_cache_file = path.join(CACHE_PATH, f'{opt["dataset"]}-{opt["sampling_method"]}-{opt["uses_graphtool"]}.cache')

if opt['cache_data'] and Path(dataset_cache_file).exists():
    print(f'Cached version of dataset found at', dataset_cache_file)
    gs = load_pickle(dataset_cache_file)
else:
    gs = create_tensors(opt['dataset'], sampling_method=opt['sampling_method'])
    if opt['cache_data']:
        print(f'Caching dataset to', dataset_cache_file)
        save_pickle(gs, dataset_cache_file)
print(f'Using {len(gs)} tensors')

cube = torch.zeros((opt['tensor_slices'], opt['slice_size'], opt['slice_size'], len(gs)))

for i in range(len(gs)):
    if opt['uses_graphtool']:
        # print(adjacency(gs[i][0]).todense().shape)
        for j in range(opt['tensor_slices']):
            #piece = adjacency(gs[i][j]).todense()
            piece = nk.algebraic.adjacencyMatrix(gs[i][j], 'dense')
            dim = piece.shape[0]
            cube[j, :dim, :dim, i] = torch.from_numpy(piece)
    else:
        ten = torch.from_numpy(gs[i])
        dim = ten.shape[1]
        cube[:, :dim, :dim, i] = ten
# print(everything[-1].size())


# In[5]:


class BasicDataset (Dataset):
    def __init__(self, data):
        self.cube = data

    def __len__(self):
        return self.cube.shape[-1]

    def __getitem__(self, idx):
        return (self.cube[:, :, :, idx], 0)


# ## Start main stuff

# In[6]:


# print('Saving input matrices')
# with open(path.join(PATH_PREFIX, 'input_mats.pkl'), 'wb') as f:
#     pickle.dump(everything, f)


# In[8]:


zoo = ModelZoo()


# In[9]:


# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

print(f"Using {opt['gen_class']} for generator")
print(f"Using {opt['disc_class']} for discriminator")

# Initialize generator and discriminator
gen_class = zoo.get_model(opt['gen_class'])
disc_class = zoo.get_model(opt['disc_class'])
discriminator = disc_class(num_nodes=opt['slice_size'], slices=opt['tensor_slices'])
generator = gen_class(
    num_nodes=opt['slice_size'], layer_size=opt['gen_layer_size'], rank=opt['rank'], num_views=opt['tensor_slices'], extra_dim=True)

discriminator.float()
generator.float()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()


# In[10]:


# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt['gen_lr'])
optimizer_D = torch.optim.RMSprop(
    discriminator.parameters(), lr=opt['disc_lr'])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# In[11]:


test_dataset = BasicDataset(cube)
sampler = RandomSampler(test_dataset)
data_loader = utils.DataLoader(
    test_dataset, batch_size=opt['batch_size'], sampler=sampler)


# In[12]:


trainer = Trainer(generator, discriminator, optimizer_G, optimizer_D, opt,
                  use_cuda=cuda, epoch_print_every=3,
                  print_every=150, checkpoint_path=SAVE_PATH,
                  critic_iterations=opt['critic_iterations'],
                  generator_iterations=opt['generator_iterations'],
                  checkpoint_every=opt['save_interval'], rank_lambda=opt['rank_lambda'],
                  penalty_type=opt['penalty_type'], wandb=wandb, rank_penalty_method=opt['rank_penalty_method'],
                  n_graph_sample_batches=opt['n_graph_sample_batches'], batch_size=opt['batch_size'],
                  eval_every=opt['eval_every'], gs=gs,
                  eval_method=opt['eval_method'], given_raw_graphs=opt['uses_graphtool'])


# In[13]:


try:
    trainer.train(data_loader, opt['n_epochs'], save_training_gif=False)
except KeyboardInterrupt:
    print('Caught keyboard interrupt, saving model...')
    torch.save(generator.state_dict(), path.join(
        SAVE_PATH, '{}_generator_v{}-final'.format(opt['model_name'], VERSION)))
    torch.save(discriminator.state_dict(), path.join(
        SAVE_PATH, '{}_discriminator_v{}-final'.format(opt['model_name'], VERSION)))
    torch.save(optimizer_G.state_dict(), path.join(
        SAVE_PATH, '{}_optimizerG_v{}-final'.format(opt['model_name'], VERSION)))
    torch.save(optimizer_D.state_dict(), path.join(
        SAVE_PATH, '{}_optimizerD_v{}-final'.format(opt['model_name'], VERSION)))
    print('Done, bye!')

print('Done, saving model...')
torch.save(generator.state_dict(), path.join(
    SAVE_PATH, '{}_generator_v{}-final'.format(opt['model_name'], VERSION)))
torch.save(discriminator.state_dict(), path.join(
    SAVE_PATH, '{}_discriminator_v{}-final'.format(opt['model_name'], VERSION)))
torch.save(optimizer_G.state_dict(), path.join(
    SAVE_PATH, '{}_optimizerG_v{}-final'.format(opt['model_name'], VERSION)))
torch.save(optimizer_D.state_dict(), path.join(
    SAVE_PATH, '{}_optimizerD_v{}-final'.format(opt['model_name'], VERSION)))
