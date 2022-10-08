'''Utility methods relating to graph loading and preprocessing.'''

from collections import deque
import random
import networkx as nx
import numpy as np

def random_binary(n, use_sparsity=True, sparsity=0.5):
    '''Generate a random binary matrix given its sparsity.'''

    if use_sparsity:
        r = np.random.uniform(size=(n, n))
        G = np.zeros((n, n))
        G[r >= sparsity] = 1
    else:
        G = np.random.randint(2, size=(n, n))
    return G


def graph_threshold(G, threshold=0.5):
    '''Perform thresholding on a non-binary adjacency matrix.'''
    G[G >= threshold] = 1
    G[G < threshold] = 0
    return G


def make_fractional(G_in, threshold=0.5, inplace=True):
    #     print(G_in.dtype)
    if inplace:
        G = G_in
    else:
        G = G_in.copy()
    num_nonzero = np.count_nonzero(G)
    num_zero = np.count_nonzero(G == 0)
    rand = np.random.uniform(threshold, 1, num_nonzero)
    rand2 = np.random.uniform(1e-8, threshold, num_zero)
    G[G != 0] = rand
    G[G == 0] = rand2
    return G

# Gets the kth slice of size slice_size of G
def graph_slice(G, k=0, slice_size=100, random_vals=False, threshold=0.5):
    s = G[k:k+slice_size, k:k+slice_size]
    if random_vals:
        s = s.copy()
        num_nonzero = np.count_nonzero(s)
        num_zero = np.count_nonzero(s == 0)
#         print(num_nonzero)
#         print('Found {} > 0 indices: '.format((num_nonzero)))
        rand = np.random.uniform(threshold, 1, num_nonzero)
        s[s != 0] = rand
        rand2 = np.random.uniform(1e-8, threshold, num_zero)
        s[s == 0] = rand2
        # optional
#         np.fill_diagonal(s, 0)
        return s
    return s


# Generate permutations of the graph
def generate_permutation(G, nodes=None, zero_index=True):
    # backwards compat for function calls that pass in a nodes variable
    if nodes is None:
        nodes = list(G.nodes())
    random.shuffle(nodes)
    num_nodes = len(nodes)
    if zero_index:
        mapping = {x: nodes[x] for x in range(num_nodes)}
    else:
        mapping = {x: nodes[x - 1] for x in range(1, num_nodes + 1)}
    return nx.relabel_nodes(G, mapping)


def hashGraph(G, buckets=100, unweighted=False, fractional=False):
    '''
    Given an unweighted graph, hash it by placing each node into a random "bucket".
    This creates a buckets x buckets sized graph.
    An edge between two buckets, b0 and b1, exists if any element in a b0 has a connection to an element in b1.
    The weight of this edge is equal to the number of edges between edges in b0 and b1.

    Suprisingly, it's faster to run this single-threaded than using multiprocessing (at least according to my tests).
    This may be due to the overhead required to copy G to multiple processes.
    '''
    n = G.shape[0]
    output = np.zeros((buckets, buckets))
    # Map of node => bucket
    table = np.random.randint(0, buckets, n)

    # Iterate over every node
    for i in range(n):
        # Find non-zero values in the adj matrix (edges)
        #   and find the bucket of the destination
        indices = table[np.nonzero(G[i, :] > 0)]
        bucketNum = table[i]  # bucket that the current node belongs to
        np.add.at(output[bucketNum, :], indices, 1)  # add to the row

        # Do the same thing, but the other way around (for directed graphs)
        indices = table[np.nonzero(G[:, i] > 0)]
        np.add.at(output[:, bucketNum], indices, 1)

    if unweighted:
        # normalize w/ min-max normalization [TODO: explore alternate methods]
        if fractional:
            output = (output - output.min()) / (output.max() - output.min())
        else:  # otherwise, just threshold
            output[output > 0] = 1

    return output


def kron_graph(seed, n=6):
    '''Generate a Kronecker graph given a starting seed.'''
    K = seed
    for i in range(n):
        K = np.kron(seed, K)
    return K

def fastHashGraph(G, buckets=100, unweighted=True):
    '''
    Hashes the graph by multiplying it with a matrix randomly filled with 0/1's
    Should be faster than the random bucket method.
    '''
    n = G.shape[0]
    B = np.random.randint(0, 2, (n, buckets))
    B[B == 0] = -1
    out = B.T @ G @ B
    if unweighted:
        out[out > 0] = 1
        out[out <= 0] = 0
    return out

def approx_rank(sing, eps = 10**-6):
    for i, s in enumerate(sing):
        if s < eps:
            return i
    return len(sing)

# def form_bfs_permutation(G, bfs_suc, start):
#     out = np.zeros(A.shape)
#     n = A.shape[0]
#     Q = deque()
#     new_order = [start]
#     Q.extend(bfs_suc[start])
#     while len(Q) > 0:
#         top = Q.pop()
#         new_order.append(top)
#         if top in bfs_suc:
#             Q.extend(bfs_suc[top])

#     new_labels = dict(zip(G.nodes(), new_order))
#     return nx.relabel_nodes(G, new_labels)

def form_permutations(G, bfs_suc, start):
    Q = deque()
    new_order = [start]
    Q.extend(bfs_suc[start])
    while len(Q) > 0:
        top = Q.pop()
        new_order.append(top)
        if top in bfs_suc:
            Q.extend(bfs_suc[top])

    new_labels = dict(zip(G.nodes(), new_order))
    return nx.relabel_nodes(G, new_labels)

def get_bfs_orderings(G):
    pi_rand = generate_permutation(G, list(G.nodes()))
    out = []
    for i in pi_rand.nodes():
        out.append(form_permutations(G, dict(nx.bfs_successors(pi_rand, i)), i))
    return out

# Converts an iterable of lists of graphs to a list of tensors.
def graphs2tens(graphs):
    tens = []
    for gs in graphs:
        tens.append(np.stack([nx.to_numpy_array(g) for g in gs], axis=-1))
    return tens
