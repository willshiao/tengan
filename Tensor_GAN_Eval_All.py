#!/usr/bin/env python
# coding: utf-8

import itertools
from karateclub.graph_embedding.graph2vec import Graph2Vec
from differ.multiview import MultiviewDiffer, TensorDiffer
import glob
import graphrnn as grnn
import json
import networkx as nx
import numpy as np
import random
import torch
import pathlib
import sys
from gutils.general import load_pickle, save_pickle
from pathlib import Path

from collections import defaultdict
from dsloader import util
from dsloader.tensor_creator import create_tensors
from ggan import ModelZoo
from os import path
from tqdm import tqdm

SAMPLE_BATCHES = 5
TS_SAMPLES = 50
TS_MAX_RANK = 40
SAMPLE_MMD_STATS = True
MMD_SAMPLE_SIZE = 100

random.seed(424242)
np.random.seed(424242)

cuda = True if torch.cuda.is_available() else False
print('CUDA is enabled' if cuda else 'CUDA is not enabled')

VERSION = '1.0.1'
MODEL_NAME = 'TGAN_football_small_r20_multigraph_rw'
USE_MMD = False
OUTPUT_DIR = 'tensor-all-eval-out'

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

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

CACHE_PATH = 'cache'
BASE_PATH = 'results/'
PATH_PREFIX = path.join(BASE_PATH, '{}_v{}'.format(MODEL_NAME, VERSION))
SAVE_PATH = path.join(PATH_PREFIX, 'models/')
IMG_PATH = path.join(PATH_PREFIX, 'images/')
STATS_PATH = path.join(PATH_PREFIX, 'stats/')


with open(path.join(PATH_PREFIX, 'params.json'), 'r') as f:
    opt = json.load(f)


print(f'Saving to {SAVE_PATH}')
print(f'Using parameters: {opt}')

if 'uses_graphtool' in opt:
    uses_graphtool = opt['uses_graphtool']
else:
    uses_graphtool = False

if 'cache_data' in opt:
    cache_data = opt['cache_data']
else:
    cache_data = True

dataset_cache_file = path.join(CACHE_PATH, f'{opt["dataset"]}-{opt["sampling_method"]}-{uses_graphtool}-raw.cache')

if cache_data and Path(dataset_cache_file).exists():
    print(f'Cached version of dataset found at', dataset_cache_file)
    gs = load_pickle(dataset_cache_file)
else:
    print(f'Creating tensors from {opt["dataset"]}')
    gs = create_tensors(opt['dataset'], sampling_method=opt['sampling_method'], get_raw=True)
    if cache_data:
        print(f'Caching dataset to', dataset_cache_file)
        save_pickle(gs, dataset_cache_file)

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

def load_wrapper(loc):
    if cuda:
        return torch.load(loc)
    print('CUDA not found, loading to CPU...')
    return torch.load(loc, map_location=torch.device('cpu'))


def load_models(fdir, G, epoch):
    if epoch == 'final':
        G.load_state_dict(load_wrapper(path.join(fdir, f'{opt["model_name"]}_generator_v{VERSION}-final')))
    else:
        G.load_state_dict(load_wrapper(path.join(fdir, f'ctgan_generator_epoch-{epoch}.model')))
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
    for k in range(opt['tensor_slices']):
        base_Gs[k].append(nx.from_numpy_array(G[k, :, :], create_using=nx.DiGraph))

# Delete gs to save memory
del gs

zipped_base_Gs = list(zip(*(base_Gs[x] for x in range(opt['tensor_slices']))))

out = []
for epoch in epochs:
    print(f'\n=========== Epoch {epoch} ===========')
    print(f'Evaluating performance at epoch {epoch}')
    generator = load_models(SAVE_PATH, generator, epoch)
    generator.eval()

    degree_stats = defaultdict(list)
    clustering_stats = defaultdict(list)
    orbit_stats = defaultdict(list)

    print('Generating samples...')
    gen_Gs = defaultdict(list)
    for i in tqdm(range(SAMPLE_BATCHES)):
        z = generator.sample_latent(opt['batch_size'])
        if cuda:
            z = z.cuda()
        res = generator(z)
        g_np = res.detach().cpu().numpy()

        for j in range(g_np.shape[0]):
            tmp_views = []
            for k in range(opt['tensor_slices']):
                tmp = g_np[j, k, :, :].copy()

                util.graph_threshold(tmp, threshold=0.1)
                graph = nx.from_numpy_array(tmp, create_using=nx.DiGraph)
                # remove_unconnected(graph)
                # f = plt.figure()
                if len(graph) > 0:
                    tmp_views.append(graph)
            if len(tmp_views) == opt['tensor_slices']:
                for k in range(len(tmp_views)):
                    gen_Gs[k].append(tmp_views[k])
            else:
                print('WARNING: skipped invalid graph')
    print(f'Generated {len(gen_Gs[0])} graphs')

    # print(self.base_Gs[k])
    # print(gen_Gs[k])

    if SAMPLE_MMD_STATS:
        sampled_base_idxs = random.sample(range(len(base_Gs[0])), MMD_SAMPLE_SIZE)
        sampled_gen_idxs = random.sample(range(len(gen_Gs[0])), MMD_SAMPLE_SIZE)

    for k in range(g_np.shape[1]):
        if SAMPLE_MMD_STATS:
            sampled_base = list(itertools.compress(base_Gs[k], sampled_base_idxs))
            sampled_gen = list(itertools.compress(gen_Gs[k], sampled_gen_idxs))
        else:
            sampled_base = base_Gs[k]
            sampled_gen = gen_Gs[k]
        print('Calulating degree stats...')
        deg_stat = grnn.degree_stats(sampled_base, sampled_gen)
        print(f'Degree difference at epoch {epoch} and slice {k}: {deg_stat}')
        degree_stats[k].append(deg_stat)

        print('Calulating clustering stats...')
        clustering_stat = grnn.clustering_stats(sampled_base, sampled_gen)
        print(f'Clustering difference at epoch {epoch} and slice {k}: {clustering_stat}')
        clustering_stats[k].append(clustering_stat)

        print('Calculating orbit stats...')
        orbit_stat = grnn.orbit_stats_all(sampled_base, sampled_gen)
        print(f'Orbit difference at epoch {epoch} and slice {k}: {orbit_stat}')
        orbit_stats[k].append(orbit_stat)

    mean_degree = np.mean([degree_stats[x] for x in range(g_np.shape[1])])
    mean_clustering = np.mean([clustering_stats[x] for x in range(g_np.shape[1])])
    mean_orbit = np.mean([orbit_stats[x] for x in range(g_np.shape[1])])

    # Do classifier-based eval
    print('Running classifier-based eval...')
    zipped_gen_Gs = list(zip(*(gen_Gs[x] for x in range(opt['tensor_slices']))))
    differ = MultiviewDiffer(zipped_base_Gs, zipped_gen_Gs, use_ensemble_model=True, embedding_model=Graph2Vec(), split_first=False, even_out=True)
    res = differ.eval()
    f1_score = res['f1']
    acc_score = res['accuracy']
    base_sample_count = len(zipped_base_Gs)
    gen_sample_count = len(zipped_gen_Gs)

    # Do tensor-based eval
    print('Running tensor-based eval...')

    differ = TensorDiffer(zipped_base_Gs, zipped_gen_Gs)
    dists, self_dists = differ.pairwise_sampled_eval(max_rank=TS_MAX_RANK, n_samples=TS_SAMPLES)
    dists_arr, self_dists_arr = np.array(dists), np.array(self_dists)

    filtered_dists = dists_arr[np.isfinite(dists_arr)]
    filtered_self_dists = self_dists_arr[np.isfinite(self_dists_arr)]
    tensor_score = np.sum(filtered_dists) / np.sum(filtered_self_dists)

    # Save every time we finish an epoch
    result = {
        'epoch': epoch,
        'mean_degree': mean_degree,
        'mean_clustering': mean_clustering,
        'mean_orbit': mean_orbit,
        'degree_stats': degree_stats,
        'orbit_stats': orbit_stats,
        'clustering_stats': clustering_stats,
        'f1_score': f1_score,
        'acc_score': acc_score,
        'base_sample_count': base_sample_count,
        'gen_sample_count': gen_sample_count,
        'ten_dists': filtered_dists.tolist(),
        'self_ten_dists': filtered_self_dists.tolist(),
        'tensor_score': tensor_score
    }
    out.append(result)

    fname = path.join(OUTPUT_DIR, f'ten_result_{MODEL_NAME}_{VERSION}.json')
    with open(fname, 'w') as f:
        print(f'Writing results to JSON... @ {fname}')
        json.dump(out, f)

# # Happens to be currently true
# epochs[-1] = epochs[-2] + 1

# plt.plot(epochs, degree_stats)
# plt.plot(epochs, clustering_stats)
# plt.plot(epochs, orbit_stats)
# plt.savefig('eval_plot.png')

print('Done!')
