# Converts TenGAN datasets to work with HGEN
from argparse import Namespace
from collections import defaultdict
from functools import cache
import sys
from dsloader.tensor_creator import create_tensors
import networkx as nx
import pathlib
import os
from os import path
from gutils.general import save_pickle
import HGEN.utils as hutils 
import hin2vec.main as hmain

HIN2VEC_CHUNK_SZ = 100
DSET_NAME = 'football_small' # suffix doesn't matter, but still needed for compatibility
full_dset_name = DSET_NAME.rsplit('_', 1)[0]
OUTPUT_DIR = f'HGEN/{full_dset_name}'

# Check for existence so we don't clobber
if pathlib.Path(OUTPUT_DIR).exists():
    print(f'ERROR: {OUTPUT_DIR} already exists!')
    sys.exit(1)

# Create new directory
os.mkdir(OUTPUT_DIR)

print(f'Loading dataset {DSET_NAME}...')
_, views = create_tensors(DSET_NAME, get_raw=True, get_single_graph=True)
n_views = 6

# First, relabel all nodes
print('Relabeling all nodes')
n_nodes = len(views[0])
# Check to make sure our view is 0-indexed
assert min(views[0].nodes()) == 0
assert max(views[0].nodes()) == n_nodes - 1
node_mapping = defaultdict(list)

for k in range(1, n_views):
    offset = n_nodes * k
    nodes = views[k].nodes()
    new_labels = [x + offset for x in nodes]
    label_dict = dict(zip(nodes, new_labels))

    # store mappings
    for old, new in label_dict.items():
        node_mapping[old].append(new)

    nx.relabel_nodes(views[k], label_dict, copy=False)
    print(f'Is view {k} connected: {nx.is_weakly_connected(views[k])}')

# Next, combine all the graphs
print('Combining all graphs')
supergraph = nx.compose_all(views)
sg_nodes = len(supergraph)
assert sg_nodes == n_views * n_nodes

# Add edges between copies of the same node across different views
for node, new_vals in node_mapping.items():
    for right in new_vals:
        # Sanity check on connections:
        assert (right - node) % n_nodes == 0  
        supergraph.add_edges_from([(node, right), (right, node)])

# Create link.dat
print('Creating link.dat')
with open(path.join(OUTPUT_DIR, 'link.dat'), 'w') as f:
    for to, fr in supergraph.edges():
        f.write(f'{to} {fr}\n')

# Create node.dat
# Format appears to be [node #] [node class label] [node class #]
print('Creating node.dat')

@cache
def node_id_to_class(node_id):
    node_class = node_id // n_nodes
    node_class_label = str(node_class)
    return node_class, node_class_label

with open(path.join(OUTPUT_DIR, 'node.dat'), 'w') as f:
    for n in supergraph.nodes():
        node_class, node_class_label = node_id_to_class(n)
        f.write(f'{n} {node_class_label} {node_class}\n')

n_chunks = max(n_views // HIN2VEC_CHUNK_SZ, 1)
sz_per_chunk = len(supergraph) // n_chunks
leftover_nodes = len(supergraph) % n_chunks

for c in range(n_chunks):
    print(f'Processing hin2vec chunk #{c}')
    chunk_left = c * sz_per_chunk
    chunk_right = min((c + 1) * sz_per_chunk, len(supergraph))
    if c == n_chunks - 1:
        chunk_right = min(len(supergraph), chunk_right + leftover_nodes)

    # Save a file in the hin2vec format
    # Format: source_node	source_class	dest_node	dest_class	edge_class
    hin2vec_file = path.join(OUTPUT_DIR, f'hin2vec_edges_chunk-{c}.txt')
    with open(hin2vec_file, 'w') as f:
        # Write header line
        f.write('#source_node\tsource_class\tdest_node\tdest_class\tedge_class\n')

        for source, dest in supergraph.edges():
            if source >= chunk_left and source < chunk_right and dest >= chunk_left and source < chunk_right:
                _, source_class = node_id_to_class(source)
                _, dest_class = node_id_to_class(dest)
                edge_class = f'{source_class}-{dest_class}'
                # source -= chunk_left
                # dest -= chunk_left
                f.write('\t'.join(str(x) for x in [source, source_class, dest, dest_class, edge_class]) + '\n')

# Run hin2vec
print('Running hin2vec')

# Create arguments
# Use defaults for everything except dim
args = Namespace()
args.walk_length = 100
args.walk_num = 10
args.neg = 5
args.dim = 32
args.alpha = 0.025
args.window = 3
args.num_processes = 28
args.allow_circle = False
args.sigmoid_reg = False

for c in range(n_chunks):
    print(f'Running hin2vec on chunk #{c}')
    output_vecs = path.join(OUTPUT_DIR, f'hin2vec_vectors_chunk-{c}.txt')
    output_mps = path.join(OUTPUT_DIR, f'hin2vec_metapaths_chunk-{c}.txt')
    hin2vec_file_path = path.join(OUTPUT_DIR, f'hin2vec_edges_chunk-{c}.txt')

    hmain.main(hin2vec_file_path, output_vecs, output_mps, args)
    print(f'Done running hin2vec on chunk #{c}')

# Convert hin2vec output into a pickle
print('Pickling hin2vec output')

output_dict = {}
for c in range(n_chunks):
    output_vecs = path.join(OUTPUT_DIR, f'hin2vec_vectors_chunk-{c}.txt')

    with open(output_vecs, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            pieces = line.strip().split(' ')
            node_num = int(pieces[0])
            output_dict[node_num] = list(map(float, pieces[1:]))
            assert len(pieces[1:]) == args.dim

save_pickle(output_dict, path.join(OUTPUT_DIR, f'hin2vec_{full_dset_name}_32.p'))
print('Successfully pickled hin2vec output!')

# Generate all paths of length 2 - 4
for path_len in range(2, 5):
    print(f'Generating paths of length {path_len}')
    paths = []
    for n in supergraph.nodes():
        for p in hutils.findAllPaths(n, lambda x: supergraph.neighbors(x), path_len):
            paths.append(p)
    save_pickle(paths, path.join(OUTPUT_DIR, f'path_len_{path_len}.p'))
    print(f'Successfully generated {len(paths)} paths of length {path_len}!')

print('Done!')
