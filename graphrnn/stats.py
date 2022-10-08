import concurrent.futures
from multiprocessing import Pool
import uuid
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess as sp
import time
from tqdm import tqdm
from pathlib import Path
import random

from . import mmd

PRINT_TIME = True

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def add_tensor(x,y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x+y

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False, max_workers=None):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref),len(sample_pred))
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed, mmd_dist)
    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True, max_workers=None):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        #total = 0
        #for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        #print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
                               sigma=1.0/10, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed, mmd_dist)
    return mmd_dist

# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
        '3path' : [1, 2],
        '4cycle' : [8],
}
COUNT_START_STR = 'orbit counts: \n'

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
#     seen = set()
    for (u, v) in G.edges():
        # Don't print self-loops and duplicate undirected edges
        if u != v:
            edges.append((id2idx[str(u)], id2idx[str(v)]))
#             seen.add((u, v))
    return edges

dir_path = os.path.dirname(os.path.realpath(__file__))
ORCA_BIN_PATH = os.path.join(dir_path, 'orca/orca')

ORCA_TMP_PATH = os.getenv('ORCA_PATH', default=os.path.join(dir_path, '../orca_tmp'))
print('Using orca path:', ORCA_TMP_PATH)

Path(ORCA_TMP_PATH).mkdir(exist_ok=True)

def orca(graph):
#     print('Running orca')
    graph = graph.to_undirected()
    tmp_fname = ORCA_TMP_PATH + '/orca_tmp_{}.txt'.format(uuid.uuid4().hex[:6])
    try:
        with open(tmp_fname, 'w') as f:
            reindexed = edge_list_reindexed(graph)
            f.write(f'{graph.number_of_nodes()} {len(reindexed)}\n')
            # f.write(str(graph.number_of_nodes()) + ' ' + str(len()) + '\n')
            for (u, v) in reindexed:
                f.write(str(u) + ' ' + str(v) + '\n')

        output = sp.check_output([ORCA_BIN_PATH, 'node', '4', tmp_fname, 'std'])
        output = output.decode('utf8').strip()
        
        idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
        output = output[idx:]
        node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
            for node_cnts in output.strip('\n').split('\n')])
    finally:
        try:
            os.remove(tmp_fname)
        except OSError as err:
            print('Warning: passed on orca: {}'.format(err))
            pass
    return node_orbit_counts
    

def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', ground_truth_match=None, bins=100):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]
    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        #hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
            is_hist=False)
    #print('-------------------------')
    #print(np.sum(total_counts_ref) / len(total_counts_ref))
    #print('...')
    #print(np.sum(total_counts_pred) / len(total_counts_pred))
    #print('-------------------------')
    return mmd_dist

def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []
 
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    # test change
    for G in graph_pred_list_remove_empty:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
            is_hist=False, sigma=30.0)

    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    return mmd_dist

def get_orbits(G):
    try:
        orbit_counts = orca(G)
    except:
        return None
    return np.sum(orbit_counts, axis=0) / G.number_of_nodes()

def orbit_stats_all_mp(graph_ref_list, graph_pred_list, processes=20, chunksize=60):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    prev = datetime.now()

    with Pool(processes) as p:
        total_counts_ref = [x for x in p.imap_unordered(get_orbits, graph_ref_list, chunksize=chunksize) if x is not None]
        total_counts_pred = [x for x in p.imap_unordered(get_orbits, graph_pred_list_remove_empty, chunksize=chunksize) if x is not None]

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
            is_hist=False, sigma=30.0)

    elapsed = datetime.now() - prev
    print('Time computing orbit mmd: ', elapsed, mmd_dist)
    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    return mmd_dist
