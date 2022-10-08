from collections import Counter
from multiprocessing import Pool
import networkx as nx
from networkx.algorithms.simple_paths import PathBuffer
import numpy as np
from os import path
#from multiprocessing import Pool
import glob
import torch
from tqdm import tqdm, trange
import seaborn as sns
import matplotlib.pyplot as plt
import random
#import graph_tool as gt
#from nx2gt.nx2gt import nx2gt
#from networkit.nxadapter import nx2nk
import networkit as nk
from karateclub.graph_embedding import Graph2Vec
import time
import pandas as pd
import pickle as pkl
import sys

POOL_SIZE = 28

def graphs2tens(graphs):
    tens = []
    for gs in tqdm(graphs):
        tens.append(np.stack([nx.to_numpy_array(g) for g in gs], axis=0))
    return tens

def load_mtx(loc, weighted=False):
    mtx_files = glob.glob(path.join(loc, '*.mtx'))
    id_file = glob.glob(path.join(loc, '*.ids'))[0]
    # filter out 500 files
    mtx_files = [f for f in mtx_files if '500.mtx' not in f]

    # create mapping
    tweet2id = {}
    cnt = 0
    with open(id_file, 'r') as f:
        for line in f:
            tweet_id = line.strip()
            tweet2id[tweet_id] = cnt
            cnt += 1

    ten = None
    for k, slice_file in enumerate(mtx_files):
        # print('Opening ', slice_file)
        with open(slice_file, 'r') as f:
            first_line = True
            for line in f:
                if first_line:
                    m, n, e = [int(x) for x in line.split(' ')]
                    if m != n:
                        print(f'Warning: m != n for {slice_file}')
                    if ten is None:
                        ten = np.zeros((m, n, len(mtx_files)))
                    first_line = False
                else:
                    i_id, j_id, w = line.split(' ')
                    w = int(w)
                    try:
                        i, j = tweet2id[i_id], tweet2id[j_id]
                    except KeyError as e:
                        print(f'Couldn\'t find ID for: ', e)
                        continue
                    if weighted:
                        ten[i, j, k] = w
                    else:
                        ten[i, j, k] = 1

        # print(nx.read_edgelist(slice_file))
        # break
    T = [nx.from_numpy_array(ten[:, :, i], create_using=nx.DiGraph) for i in range(ten.shape[2])]
    return (ten, T)
    # print('Creating tensor with ')
    # print(mtx_files)

def load_from_csv(loc, weighted=False, graph_class=nx.DiGraph):
    info = []
    with open(path.join(loc, 'edges.csv'), 'r') as f:
        max_node = 0
        max_layer = 0

        for line in f:
            if line.startswith('#'):
                continue
            data = [int(x) for x in line.strip().split(',')]
            info.append(data)
            source, target, weight, layer = data
            max_node = max(source, target, max_node)
            max_layer = max(layer, max_layer)

    ten = np.zeros((max_node+1, max_node+1, max_layer))
    for source, target, weight, layer in info:
        ten[source, target, layer-1] = weight

    T = [nx.from_numpy_array(ten[:, :, i], create_using=graph_class) for i in range(ten.shape[2])]
    return (ten, T)

shared_ten_data = None

def torch_to_nx_graph(args):
    graph_axis, i, graph_class = args
    return nx.from_numpy_array(shared_ten_data.select(graph_axis, i).to_dense().numpy(), create_using=graph_class)

def load_from_tns(loc, max_size=None, weighted=False, graph_class=nx.DiGraph, graph_axis=1):
    global shared_ten_data
    # with open(loc, 'r') as f:
    #     for line in f:
    #         pieces = line.strip().split(' ')
    #         idxs = [int(x) for x in pieces[:3]]
    #         if weighted:
    #             weight = float(pieces[-1])
    #         else:
    #             weight = 1
    print(f'Reading {loc} file...')
    idxs = []
    values = []
    with open(loc, 'r') as f:
        for line in tqdm(f):
            pieces = line.strip().split(' ')
            int_idxs = [int(x) for x in pieces[:3]]

            if max_size is not None:
                size_check = [x >= max_size for x in int_idxs]
                if all(size_check):
                    break
                elif any(size_check):
                    continue
            idxs.append(torch.tensor(int_idxs))

            if weighted:
                values.append([float(pieces[-1])])
            else:
                values.append(1)

    idx_ten = torch.vstack(idxs).T
    print('idx: ', idx_ten.shape)
    value_ten = torch.tensor(values)
    print('values: ', value_ten.shape)
    ten = torch.sparse_coo_tensor(idx_ten, value_ten, size=(max_size, max_size, max_size))
    print('ten: ', ten.shape)
    print('Converting to nx.Graph:')
    shared_ten_data = ten

    with Pool(POOL_SIZE) as p:
        T = []
        parts = ((graph_axis, x, graph_class) for x in range(ten.shape[graph_axis]))
        for g in tqdm(p.imap_unordered(torch_to_nx_graph, parts, chunksize=25), total=ten.shape[graph_axis]):
            T.append(g)
    return (ten, T)


def load_from_np(loc, graph_class=nx.DiGraph, graph_axis=1, has_extra_dim=False):
    print(f'Reading {loc} file...')
    ten = np.load(loc)
    ten = np.reshape(ten, ten.shape[:3])

    print('Converting to nx.Graph:')
    print(ten.shape)
    T = [nx.from_numpy_array(ten.take(i, axis=graph_axis), create_using=graph_class) for i in trange(ten.shape[graph_axis])]
    return (ten, T)
def csr2nk(A):
    N = A.shape[0]
    Gnew = nk.graph.Graph(n=N, directed=True)
    (r, c) = A.nonzero()
    for i in range(len(r)):
        Gnew.addEdge(r[i], c[i])
    return Gnew


def create_tensor(dset_name, path_prefix='data/', use_gtools=False):
    if dset_name == 'football':
        data_dir = path.join(path_prefix, 'football')
        ten, T = load_mtx(data_dir)
    elif dset_name == 'nell2':
        file_path = path.join(path_prefix, 'nell2', 'nell2.tns')
        ten, T = load_from_tns(file_path, max_size=5000, graph_axis=1)
    elif dset_name == 'nell2_sampled':
        file_path = path.join(path_prefix, 'nell2', 'nell_compressed_1000x4x1000.np.npy')
        ten, T = load_from_np(file_path, graph_axis=1)
        ten = np.swapaxes(ten, 1, 2) # swap so last dim is # of views
    elif dset_name == 'enron_mid':
        file_path = path.join(path_prefix, 'enron', 'enron_compressed_1000x1000x10.np.npy')
        ten, T = load_from_np(file_path, graph_axis=2)
    elif dset_name == 'comm':
        data = []
        filenameFormat = 'fill this in with the right local path'
        for d in range(2, 9):
            for p in ['22', '23', '443', '445', '80']:
                print('loading day %i, port %s'%(d-1, p))
                with open(filenameFormat%(d, p), 'rb') as f:
                    data.append(pkl.load(f))
        # get vertex labels
        vertices = [d.get_vertex_dataframe() for d in data]

        # give each vertex a consistent index across views
        nodeMap = []
        for i in range(len(vertices)):
            print('creating name map '+str(i))
            tempMap = {}
            for j in range(len(vertices[i])):
                tempMap[j] = vertices[i]['name'][j]
            nodeMap.append(tempMap)

        # create a list of all names across views
        F = pd.concat(vertices)
        allNames = list(F['name'].unique())

        # make graphs and label the nodes
        T = [d.to_networkx() for d in data]
        for i in range(len(T)):
            print('relabeling nodes in graph '+str(i))
            nx.relabel_nodes(T[i], nodeMap[i], copy=False)

        # make the tensor a list of adjacency matrices
        ten = [nx.adjacency_matrix(t, nodelist=allNames) for t in T]
    else:
        raise Exception(f'Unknown dataset: {dset_name}')


    if use_gtools:
        print('converting to networkit')
        #T = [nx2gt(x) for x in T]
        T = [csr2nk(x) for x in ten]
    return ten, T

def sample_subten(tensor, graphs, min_nodes=50, max_nodes=200):
    chosen_nodes = set()
    o = tensor.shape[2]
    print('Sampling from subtensor...')
    for i in tqdm(range(o)):
        for j in graphs[i].nodes():
            sub = nx.ego_graph(graphs[i], j, radius=2)
            sz = sub.order()
            if sz < min_nodes or sz > max_nodes:
                continue
            # Dedupe any node sets
            chosen_nodes.add(tuple(sorted(sub.nodes())))

    subgraphs = []
    subtens = []
    print('Generating subgraphs...')
    for chosen in tqdm(chosen_nodes):
        sgs = [nx.DiGraph(graphs[i].subgraph(chosen)) for i in range(o)]
        subgraphs.append(sgs)
        subtens.append(np.stack([nx.to_numpy_array(sg) for sg in sgs], axis=0))
    return (subtens, subgraphs)
    # print(f'{len(chosen_nodes)} chosen node sets')
    # print('Subgraphs: ', subgraphs)
    # print('Subten: ', subtens[0].shape)

def generate_perms(sgs, n=50):
    out = []
    nodes = list(sgs[0].nodes())
    for _ in range(n):
        tmp = []
        random.shuffle(nodes)
        mapping = {x: nodes[x] for x in range(len(nodes))}
        for sg in sgs:
            tmp.append(nx.relabel_nodes(sg, mapping))
        out.append(tmp)
    return out

def permute_subgraphs(subgraphs):

    out = []
    with Pool(POOL_SIZE) as p:
        for l in tqdm(p.imap_unordered(generate_perms, subgraphs, chunksize=5), total=len(subgraphs)):
            for k in l:
                out.append(k)
            
        # for sgs in tqdm(subgraphs):
        #     nodes = list(sgs[0].nodes())
        #     for _ in range(n):
        #         tmp = []
        #         random.shuffle(nodes)
        #         mapping = {x: nodes[x] for x in range(len(nodes))}
        #         for sg in sgs:
        #             tmp.append(nx.relabel_nodes(sg, mapping))
        #         out.append(tmp)
    return out

PATCHED_PREFIX = 'patched_'
GTOOLS_SUFFIX = '_gtools'
def create_tensors(dataset='football_small', path_prefix='data/', get_single_graph=False, get_raw=False, get_raw_graphs=False, sampling_method='egonets', **kwargs):
    dataset_name, dataset_size = dataset.rsplit('_', 1)
    use_gtools = False

    if dataset_name.endswith(GTOOLS_SUFFIX):
        dataset_name = dataset_name[:len(dataset_name) - len(GTOOLS_SUFFIX)]
        use_gtools = True

    max_steps = 10000
    if dataset_size == 'small':
        min_nodes, max_nodes = 20, 50
    else:
        print(f'Error: unknown dataset size: {dataset_size}')
        return None

    (ten, T) = create_tensor(dataset_name, use_gtools=use_gtools, path_prefix=path_prefix)
    if get_single_graph:
        return ten, T

    # Allow for legacy name of "random"
    if sampling_method == 'egonets' or sampling_method == 'random':
        subtens, subgraphs = sample_subten(ten, T, min_nodes=min_nodes, max_nodes=max_nodes)
    elif sampling_method == 'random_walk':
        subtens, subgraphs = sample_subten_rw(ten, T, min_nodes=min_nodes, max_nodes=max_nodes, max_steps=max_steps, use_gtools=use_gtools, **kwargs)
    elif sampling_method == 'multigraph_rw':
        subtens, subgraphs = sample_multigraph_rw(ten, T, min_nodes=min_nodes, max_nodes=max_nodes, max_steps=max_steps, **kwargs)
    elif sampling_method == 'multigraph_rw_no_induce':
        subtens, subgraphs = sample_multigraph_rw(ten, T, min_nodes=min_nodes, max_nodes=max_nodes, use_induced_sg=False, max_steps=max_steps, **kwargs)
    elif sampling_method == 'multilayer_rw':
        subtens, subgraphs = sample_multilayer_rw(ten, T, min_nodes=min_nodes, max_nodes=max_nodes, max_steps=max_steps, **kwargs)
    elif sampling_method == 'multilayer_rw_no_induce':
        subtens, subgraphs = sample_multilayer_rw(ten, T, min_nodes=min_nodes, max_nodes=max_nodes, use_induced_sg=False, max_steps=max_steps, **kwargs)
    else:
        print('ERROR: unknown sampling method')
        return None
    print('returned from sampling')

    if get_raw and not use_gtools:
        return subtens
    elif get_raw and use_gtools:
        # convert to subtens
        print('Converting NetworKit graphs to tensors...')
        subtens = []
        for gs in tqdm(subgraphs):
            subtens.append([nk.algebraic.adjacencyMatrix(gs[k], matrixType='dense') for k in range(gs)])
        return subtens
    if get_raw_graphs:
        return subgraphs

    if not use_gtools:
        print('subgraphs: ', len(subgraphs))
        if subgraphs:
            print('subgraph type: ', type(subgraphs[0][0]))

        print('Generating subgraph permutations...')
        perm_subgraphs = permute_subgraphs(subgraphs)
        print('after perm_subgraphs: ', len(perm_subgraphs))

        print('Converting graphs to tensors...')
        perm_tens = graphs2tens(perm_subgraphs)
        print('after graphs2tens: ', len(perm_tens))

        return perm_tens
    else:
        return subgraphs

def sample_subten_rw(tensor, graphs, use_gtools=False, **kwargs):
    print('Use gtools: ', use_gtools)
    if use_gtools:
        return gtools_sample_subten_rw(tensor, graphs, **kwargs)
    return nx_sample_subten_rw(tensor, graphs, **kwargs)

# A little hack to take advantage of the fact that module-level variables are shared in Python
# This allows us to share read-only data between processes on fork()

#shared_graph_data = None
#shared_sizes = None
#
#def chosen_filter_helper(chosen):
#    max_sz = np.max(shared_sizes)
#    filt = np.zeros(max_sz)
#    for i in chosen:
#        filt[i] = 1
#    return [gt.GraphView(shared_graph_data[i], vfilt=filt[:shared_sizes[i]]) for i in range(len(shared_graph_data))]
#
#def chosen_filter_helper(chosen):
#    return [nk.graphtools.subgraphFromNodes(shared_graph_data[i], chosen) for i in range(len(shared_graph_data))]

def gtools_sample_subten_rw(tensor, graphs, min_nodes=50, max_nodes=200, max_steps=10000):
    chosen_nodes = set()
    o = len(graphs)
    sizes = []
    #global shared_graph_data
    #global shared_sizes
    #shared_graph_data = graphs

    print('Sampling from gtools subtensor...')
    for i in tqdm(range(o)):
        print('graph '+str(i))
        #verts = graphs[i].get_vertices()
        verts = [v for v in graphs[i].iterNodes() if (graphs[i].degreeOut(v) > 0)]
        print('number of nodes: '+str(len(verts)))
        np.random.shuffle(verts)
        #verts = list(graphs[i].forNodesInRandomOrder())
        sizes.append(len(verts))

        layerCtr = 0
        for j in verts:
            subset = {j}
            v = j
            ctr = 0
            while (len(subset) < max_nodes) and (ctr < max_steps):
                #N = graphs[i].get_out_neighbors(v)
                N = list(graphs[i].iterNeighbors(v))
                if len(N) == 0:
                    break
                v = np.random.choice(N)
                subset.add(v)
                ctr += 1
            # sub = nx.subgraph(graphs[i], subset)
            sz = len(subset)
            if sz < min_nodes or sz > max_nodes:
                continue
            else:
                layerCtr += 1
            # Dedupe any node sets
            chosen_nodes.add(tuple(sorted(subset)))
            if layerCtr >= 10000:
                break


    #shared_sizes = sizes
    subgraphs = []
    print('Generating subgraphs...')
    print('size of graph: '+str(sys.getsizeof(graphs)))
    #with Pool(28) as p:
    #    for val in tqdm(p.imap_unordered(chosen_filter_helper, chosen_nodes, chunksize=25)):
    #        subgraphs.append(val)
    chosen_nodes = list(chosen_nodes)
    for j in tqdm(range(len(chosen_nodes))):
        newList = []
        for i in range(o):
            sg = nk.graphtools.subgraphFromNodes(graphs[i], chosen_nodes[j], compact=True)
            newList.append(sg)
        #subgraphs.append([nk.graphtools.subgraphFromNodes(graphs[i], chosen) for i in range(len(graphs))])
        subgraphs.append(newList)
    
    return None, subgraphs

def nx_sample_subten_rw(tensor, graphs, min_nodes=50, max_nodes=200, max_steps=10000):
    filtered_sizes = []
    chosen_nodes = set()
    o = tensor.shape[2]
    print('Sampling from subtensor...')
    for i in tqdm(range(o)):
        for j in graphs[i].nodes():
            #sub = nx.ego_graph(graphs[i], j, radius=2)  #replacing this with random walk code
            subset = {j}
            v = j
            ctr = 0
            while (len(subset) < max_nodes) and (ctr < max_steps):
                N = list(nx.neighbors(graphs[i], v))
                if len(N)==0:
                    break
                v = np.random.choice(N)
                subset.add(v)
                ctr += 1
            sub = nx.subgraph(graphs[i], subset)

            sz = sub.order()
            if sz < min_nodes or sz > max_nodes:
                # print('Size does not meet cutoff: ', sz)
                filtered_sizes.append(sz)
                continue
            # Dedupe any node sets
            chosen_nodes.add(tuple(sorted(sub.nodes())))

    print('Filtered subgraphs with of size with frequencies: ', Counter(filtered_sizes))
    subgraphs = []
    subtens = []
    print('Generating subgraphs...')
    for chosen in tqdm(chosen_nodes):
        sgs = [nx.DiGraph(graphs[i].subgraph(chosen)) for i in range(o)]
        subgraphs.append(sgs)
        subtens.append(np.stack([nx.to_numpy_array(sg) for sg in sgs], axis=0))
    return (subtens, subgraphs)
    # print(f'{len(chosen_nodes)} chosen node sets')
    # print('Subgraphs: ', subgraphs)
    # print('Subten: ', subtens[0].shape)

def sample_multigraph_rw(tensor, graphs, min_nodes=50, max_nodes=200, max_steps=10000, use_induced_sg=True):
    chosen_nodes = list()
    chosen_edges = list()
    o = tensor.shape[2]
    filtered_sizes = []

    print('Creating multigraph')
    G = nx.MultiDiGraph()
    for i in tqdm(range(o)):
        G1 = graphs[i].copy()
        G.add_edges_from([(e[0], e[1], i) for e in G1.edges()])
    E = dict()
    for e in G.edges:
        if e[0] not in E:
            E[e[0]] = []
        E[e[0]].append(e)

    print('Sampling from subtensor...')
    for j in G.nodes():
        subset = {j}
        edges = set()
        v = j
        ctr = 0
        while (len(subset) < max_nodes) and (ctr < max_steps):
            if v not in E:
                break
            E1 = E[v]
            e_ind = np.random.choice(len(E1))
            e = E1[e_ind]
            v = e[1]
            subset.add(v)
            edges.add(e)
            ctr += 1
        sz = len(subset)
        if sz < min_nodes or sz > max_nodes:
            filtered_sizes.append(sz)
            continue
        chosen_nodes.append(tuple(sorted(subset)))
        chosen_edges.append(list(edges))

    print('Filtered subgraphs with of size with frequencies: ', Counter(filtered_sizes))
    subgraphs = []
    subtens = []
    print('Generating subgraphs...')
    for j in tqdm(range(len(chosen_nodes))):
        chosen = chosen_nodes[j]
        if use_induced_sg:
            sgs = [nx.DiGraph(graphs[i].subgraph(chosen)) for i in range(o)]
        else:
            sgs = []
            for i in range(o):
                temp = nx.DiGraph()
                temp.add_nodexamplees_from(chosen)
                sgs.append(temp)
            for e in chosen_edges[j]:
                tens_ind = e[2]
                u = e[0]
                v = e[1]
                sgs[tens_ind].add_edge(u, v)
        subgraphs.append(sgs)
        subtens.append(np.stack([nx.to_numpy_array(sg) for sg in sgs], axis=0))
    return (subtens, subgraphs)

def sample_multilayer_rw(tensor, graphs, min_nodes=50, max_nodes=200, max_steps=10000, \
                         use_induced_sg=True, p_switch=0.1):
    chosen_nodes = list()
    chosen_edges = list()
    o = tensor.shape[2]

    print('Sampling from subtensor...')
    for i in tqdm(range(o)):
        for j in graphs[i].nodes():
            subset = {j}
            edges = set()
            v = j
            layer = i
            ctr = 0
            while (len(subset) < max_nodes) and (ctr < max_steps):
                if np.random.rand() < p_switch:
                    L = set(range(o))
                    L.remove(layer)
                    layer = np.random.choice(list(L))
                    continue
                N = list(nx.neighbors(graphs[layer], v))
                if len(N) > 0:
                    v_old = v
                    v = np.random.choice(N)
                    subset.add(v)
                    edges.add((v_old, v, layer))
                ctr += 1
            sz = len(subset)
            if sz < min_nodes or sz > max_nodes:
                print('Size does not meet cutoff:', sz)
                continue
            chosen_nodes.append(tuple(sorted(subset)))
            chosen_edges.append(list(edges))

    subgraphs = []
    subtens = []
    print('Generating subgraphs...')
    for j in tqdm(range(len(chosen_nodes))):
        chosen = chosen_nodes[j]
        if use_induced_sg:
            sgs = [nx.DiGraph(graphs[i].subgraph(chosen)) for i in range(o)]
        else:
            sgs = []
            for i in range(o):
                temp = nx.DiGraph()
                temp.add_nodes_from(chosen)
                sgs.append(temp)
            for e in chosen_edges[j]:
                tens_ind = e[2]
                u = e[0]
                v = e[1]
                sgs[tens_ind].add_edge(u, v)
        subgraphs.append(sgs)
        subtens.append(np.stack([nx.to_numpy_array(sg) for sg in sgs], axis=0))
    return (subtens, subgraphs)

# import ..differ
# import repackage
# repackage.up()
# from differ.multiview import DeepWalkWrapper, MultiviewDiffer

if __name__ == '__main__':
    print('Hi')
    start = time.time()
    print(create_tensors('nell2_huge', sampling_method='multilayer_rw'))
    end = time.time()
    print(end - start)

    # print('Hi')
    # start = time.time()
    # print(create_tensors('football_gtools_small', sampling_method='random_walk'))
    # end = time.time()
    # print(end - start)

    # (ten, T) = create_tensor('football')
    # (ten, T) = create_tensor('eu_airlines')
    # # print(ten.shape)
    # # print(ten.sum())
    # # ego_sizes = []
    # # for i in range(len(T)):
    # #     for n in tqdm(T[i].nodes()):
    # #         sub = nx.ego_graph(T[i], n, radius=2)
    # #         ego_sizes.append(sub.order())
    # print('Sampling...')
    # subtens, subgraphs = sample_subten(ten, T, min_nodes=20, max_nodes=50)
    # # subtens, subgraphs = sample_subten(ten, T, min_nodes=150, max_nodes=200)
    # print(len(subtens))
    # # print(subgraphs)
    # perm_subgraphs = permute_subgraphs(subgraphs)
    # print(len(perm_subgraphs))
    # with false: {'f1': 0.5474701534963047, 'accuracy': 0.47389292795769994}
    # diff = MultiviewDiffer(perm_subgraphs, perm_subgraphs, use_ensemble_model=True, embedding_model=DeepWalkWrapper())
    # print(diff.eval())
    # p2
    # perm_tens = graphs2tens(perm_subgraphs)
    # print(len(perm_tens ))
    # print(sum(ego_sizes) / len(ego_sizes))
    # print(T[0].)
