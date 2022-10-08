import json
from os import path
import repackage
repackage.up()
from collections import defaultdict
import itertools
import random
import numpy as np
from gutils.general import load_pickle
import networkx as nx
from karateclub.graph_embedding.graph2vec import Graph2Vec
# from joblib import Memory

from differ.multiview import MultiviewDiffer, RescalDiffer, TensorDiffer
import graphrnn as grnn
from dsloader.tensor_creator import sample_subten_rw, create_tensors

# change params as needed
DATASET = 'nell2_sampled_small'
MMD_SAMPLE_SIZE = 100
TS_SAMPLES = 50
TS_MAX_RANK = 40

# # memory = Memory('job_cache')
gen_adj = load_pickle('sample_syn_graph_nell2_sampled.pkl')
print(type(gen_adj))
gen_G = nx.from_numpy_array(gen_adj, create_using=nx.DiGraph)

N_LAYERS = 6
layer_size = len(gen_G) // N_LAYERS
print('Size: ', len(gen_G), '; Layer size:', layer_size, 'Leftovers: ', len(gen_G) % layer_size)

slices = []
ten_slices = []
for k in range(N_LAYERS):
    curr_slice = gen_G.subgraph(range(k * layer_size, (k + 1) * layer_size))
    curr_slice = nx.relabel_nodes(curr_slice, {x: i for i, x in enumerate(curr_slice.nodes())})
    # print(curr_slice)
    ten_slice = nx.to_numpy_array(curr_slice)
    ten_slices.append(ten_slice)
    slices.append(curr_slice)

ten = np.dstack(ten_slices)

sampled_tens, gen_graphs = sample_subten_rw(ten, slices, min_nodes=20, max_nodes=50)

base_graphs = create_tensors(dataset=DATASET, path_prefix='../data', get_raw_graphs=True, sampling_method='multigraph_rw')
print(f'Clamping view size from {len(base_graphs[0])} to {N_LAYERS}')
base_graphs = [[x[i] for i in range(N_LAYERS)] for x in base_graphs]

gen_Gs = defaultdict(list)
base_Gs = defaultdict(list)

sampled_base_idxs = random.sample(range(len(base_graphs)), MMD_SAMPLE_SIZE)
sampled_gen_idxs = random.sample(range(len(gen_graphs)), MMD_SAMPLE_SIZE)

for b_idx in sampled_base_idxs:
    for view_num, view in enumerate(base_graphs[b_idx]):
        base_Gs[view_num].append(view)

for g_idx in sampled_gen_idxs:
    for view_num, view in enumerate(gen_graphs[g_idx]):
        gen_Gs[view_num].append(view)

# degree_stats = dict()
# clustering_stats = dict()
# orbit_stats = dict()

# for k in range(N_LAYERS):
#     sampled_base = base_Gs[k]
#     sampled_gen = gen_Gs[k]

#     print('Calulating degree stats...')
#     deg_stat = grnn.degree_stats(sampled_base, sampled_gen)
#     print(f'Degree difference at slice {k}: {deg_stat}')
#     degree_stats[k] = deg_stat

#     print('Calulating clustering stats...')
#     clustering_stat = grnn.clustering_stats(sampled_base, sampled_gen)
#     print(f'Clustering difference at slice {k}: {clustering_stat}')
#     clustering_stats[k] = clustering_stat

#     print('Calculating orbit stats...')
#     orbit_stat = grnn.orbit_stats_all(sampled_base, sampled_gen)
#     print(f'Orbit difference at slice {k}: {orbit_stat}')
#     orbit_stats[k] = orbit_stat

# mean_degree = np.mean([degree_stats[x] for x in range(N_LAYERS)])
# mean_clustering = np.mean([clustering_stats[x] for x in range(N_LAYERS)])
# mean_orbit = np.mean([orbit_stats[x] for x in range(N_LAYERS)])

mean_degree = mean_clustering = mean_orbit = degree_stats = orbit_stats = clustering_stats = f1_score = acc_score = base_sample_count = gen_sample_count = 0

# Do classifier-based eval
# print('Running classifier-based eval...')
# zipped_gen_Gs = list(zip(*(gen_Gs[x] for x in range(N_LAYERS))))
# differ = MultiviewDiffer(base_graphs, gen_graphs, use_ensemble_model=True, embedding_model=Graph2Vec(), split_first=False, even_out=True)
# res = differ.eval()
# f1_score = res['f1']
# acc_score = res['accuracy']
# base_sample_count = len(base_graphs)
# gen_sample_count = len(gen_graphs)

# Do tensor-based eval
print('Running tensor-based eval...')

differ = RescalDiffer(base_graphs, gen_graphs)
dists, self_dists = differ.pairwise_sampled_eval(max_rank=TS_MAX_RANK, n_samples=TS_SAMPLES)
dists_arr, self_dists_arr = np.array(dists), np.array(self_dists)

filtered_dists = dists_arr[np.isfinite(dists_arr)]
filtered_self_dists = self_dists_arr[np.isfinite(self_dists_arr)]
tensor_score = np.sum(filtered_dists) / np.sum(filtered_self_dists)

result = {
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

fname = path.join(f'hgen_result_{DATASET}.json')
with open(fname, 'w') as f:
    print(f'Writing results to JSON... @ {fname}')
    json.dump(result, f)
