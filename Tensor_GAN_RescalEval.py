#!/usr/bin/env python
# coding: utf-8

from karateclub.graph_embedding.graph2vec import Graph2Vec
from differ.multiview import DeepWalkWrapper, MultiviewDiffer, RescalDiffer, TensorDiffer
import dsloader
import glob
import graphrnn as grnn
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import pathlib
import sys

from collections import defaultdict, deque
from dsloader import util
from dsloader.graphrnn_creator import create
from dsloader.tensor_creator import create_tensors
from ggan import ModelZoo
import pickle
from os import path
from tqdm import tqdm

# with open('input_mats.pickle', 'rb') as f:
#     graphs = pickle.load(f)

random.seed(424242)
np.random.seed(424242)

cuda = True if torch.cuda.is_available() else False
print('CUDA is enabled' if cuda else 'CUDA is not enabled')

VERSION = '1.0.1'
MODEL_NAME = 'TGAN_football_small_r20_multigraph_rw'
USE_MMD = False
OUTPUT_DIR = 'tensor-eval-out'

pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

argc = len(sys.argv)
if argc > 1:
    VERSION = sys.argv[1]

if argc > 2:
    MODEL_NAME = sys.argv[2]
print(f'Using version: {VERSION}')

if argc > 3:
    if sys.argv[3] == 'all':
        epoch_filter = []
    else:
        epoch_filter = [int(x) if x != 'final' else x for x in sys.argv[3].split(',')]
else:
    epoch_filter = []

# Operation filters
do_degree = do_clustering = do_orbit = False
if argc > 4:
    pieces = [x.strip() for x in sys.argv[4].split(',')]
    if 'orbit' in pieces:
        do_orbit = True
    if 'degree' in pieces:
        do_degree = True
    if 'clustering' in pieces:
        do_clustering = True
    print(f'Doing {", ".join(pieces)} operations')
else:
    do_degree = do_clustering = do_orbit = True

BASE_PATH = 'results/'
PATH_PREFIX = path.join(BASE_PATH, '{}_v{}'.format(MODEL_NAME, VERSION))
SAVE_PATH = path.join(PATH_PREFIX, 'models/')
IMG_PATH = path.join(PATH_PREFIX, 'images/')
STATS_PATH = path.join(PATH_PREFIX, 'stats/')


with open(path.join(PATH_PREFIX, 'params.json'), 'r') as f:
    opt = json.load(f)
n_slices = min(opt['tensor_slices'], 4)

print(f'Saving to {SAVE_PATH}')
print(f'Using parameters: {opt}')

print(f'Creating tensors from {opt["dataset"]}')
gs = create_tensors(opt['dataset'], get_raw=True, sampling_method=opt['sampling_method'])
print(f'Using {len(gs)} tensors')


def get_epochs(base_path):
    files = glob.glob1(base_path, '*.model')
    return sorted(set([int(x.rsplit('_', 1)[1].split('-')[1].split('.')[0]) for x in files]))


epochs = get_epochs(SAVE_PATH)
print(f'Found epochs: {epochs}')
if epoch_filter:
    epochs = epoch_filter
else:
    # Add last epoch
    epochs += ['final']
print(f'Using epochs: {epoch_filter}')

zoo = ModelZoo()
gen_class = zoo.get_model(opt['gen_class'])

# Initialize generator
generator = gen_class(num_nodes=opt['slice_size'], layer_size=opt['gen_layer_size'], rank=opt['rank'], num_views=opt['tensor_slices'], extra_dim=True)
generator.float()

if cuda:
    generator.cuda()

def load_models(fdir, G, epoch):
    if epoch == 'final':
        G.load_state_dict(torch.load(path.join(fdir, f'{opt["model_name"]}_generator_v{VERSION}-final')))
    else:
        G.load_state_dict(torch.load(path.join(fdir, f'ctgan_generator_epoch-{epoch}.model')))
    return G

def remove_unconnected(G):
    # to_remove = []
    # for node in G.nodes():
    #     if len(G.edges(node)) == 0:
    #         to_remove.append(node)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

base_Gs = defaultdict(list)
for G in gs:
    for k in range(n_slices):
        base_Gs[k].append(nx.from_numpy_array(G[k, :, :], create_using=nx.DiGraph))

degree_stats = defaultdict(list)
clustering_stats = defaultdict(list)
orbit_stats = defaultdict(list)
f1_scores = []
acc_scores = []
base_sample_count = []
gen_sample_count = []
all_dists = []
all_self_dists = []
all_scores = []

for epoch in epochs:
    print(f'\n=========== Epoch {epoch} ===========')
    print(f'Evaluating performance at epoch {epoch}')
    generator = load_models(SAVE_PATH, generator, epoch)
    generator.eval()

    print('Generating samples...')
    gen_Gs = defaultdict(list)
    for i in tqdm(range(20)):
        z = generator.sample_latent(opt['batch_size']).cuda()
        res = generator(z)
        g_np = res.detach().cpu().numpy()

        for j in range(n_slices):
            tmp_views = []
            for k in range(n_slices):
                tmp = g_np[j, k, :, :].copy()

                util.graph_threshold(tmp, threshold=0.0001)
                graph = nx.from_numpy_array(tmp, create_using=nx.DiGraph)
                # remove_unconnected(graph)
                # f = plt.figure()
                if len(graph) > 0:
                    tmp_views.append(graph)
            if len(tmp_views) == n_slices:
                for k in range(len(tmp_views)):
                    gen_Gs[k].append(tmp_views[k])
            else:
                print('WARNING: skipped invalid graph')
    print(f'Generated {len(gen_Gs[0])} graphs')

    if len(gen_Gs) == 0:
        all_dists.append([])
        all_scores.append([])
        all_self_dists.append([])
        print(f'Skipped epoch {epoch}')
        continue
    zipped_base_Gs = list(zip(*(base_Gs[x] for x in range(n_slices))))
    zipped_gen_Gs = list(zip(*(gen_Gs[x] for x in range(n_slices))))
    print('profile:', len(zipped_base_Gs), len(zipped_base_Gs[0]), len(zipped_base_Gs[0][0]))
    # print('base_Gs stats:')
    # print(len(base_Gs))
    # print(base_Gs.shape)

    differ = RescalDiffer(zipped_base_Gs, zipped_gen_Gs)
    dists, self_dists = differ.pairwise_sampled_eval(max_rank=40, n_samples=50)
    dists_arr, self_dists_arr = np.array(dists), np.array(self_dists)
    all_dists.append(dists)
    all_self_dists.append(self_dists)
    filtered_dists = dists_arr[np.isfinite(dists_arr)]
    filtered_self_dists = self_dists_arr[np.isfinite(self_dists_arr)]
    all_scores.append(np.sum(filtered_dists) / np.sum(filtered_self_dists))

    result = {
        'epochs': epochs,
        'scores': all_scores,
        'dists': all_dists,
        'self_dists': all_self_dists
    }

    fname = path.join(OUTPUT_DIR, f'rten_result_{MODEL_NAME}_{VERSION}.json')
    with open(fname, 'w') as f:
        print(f'Writing results to JSON... @ {fname}')
        json.dump(result, f)

# # Happens to be currently true
# epochs[-1] = epochs[-2] + 1

# plt.plot(epochs, degree_stats)
# plt.plot(epochs, clustering_stats)
# plt.plot(epochs, orbit_stats)
# plt.savefig('eval_plot.png')

print('Done!')
