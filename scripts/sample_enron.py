from graph_tool import value_types
import repackage
repackage.up()

import numpy as np
import dsloader.tensor_creator as tc
from os import path
from tqdm import tqdm
import torch
from numba.core import types
from numba.typed import Dict
from numba import njit
import mmap
from pathlib import Path

path_prefix = '../data/enron'
loc = path.join(path_prefix, 'enron.tns')
weighted = True
cached_idx_path = path.join(path_prefix, 'cached_idxs.pt')
cached_val_path = path.join(path_prefix, 'cached_vals.pt')

USE_CACHE = True
SAMPLES_I = 1000
SAMPLES_J = 1000
AGG_SIZE_K = 60
SAMPLES_K = 10

def calc_sparsity(sparse_ten):
    n_vals = sparse_ten.values().shape[0]
    total_vals = torch.numel(sparse_ten)
    return n_vals / total_vals

# den = torch.rand((101, 102, 103))
# ten = den.to_sparse()

# print(ten.index)

print(f'Reading {loc} file...')
idxs = []
values = []

if Path(cached_idx_path).exists() and Path(cached_val_path).exists() and USE_CACHE:
    print('Cached files found, loading from cache...')
    idx_ten = torch.load(cached_idx_path)
    value_ten = torch.load(cached_val_path)
    ten = torch.sparse_coo_tensor(idx_ten, value_ten)
    print('ten: ', ten.shape)

    print('Coalescing tensor...')
    ten = ten.coalesce()
else:
    print('No cached values found, loading from scratch...')
    with open(loc, 'r') as f_obj:
        with mmap.mmap(f_obj.fileno(), length=0, access=mmap.ACCESS_READ) as f:
            with tqdm() as pbar:
                while True:
                    line = str(f.readline(), 'utf8')
                    if line == '': break
                    pbar.update(1)

                    pieces = line.strip().split(' ')
                    int_idxs = [int(x) for x in pieces[:4]]
                    # exclude words dim
                    idxs.append(torch.tensor([int_idxs[0], int_idxs[1], int_idxs[3]]).view(3, 1))

                    if weighted:
                        values.append([float(pieces[-1])])
                    else:
                        values.append(1)

    idx_ten = torch.hstack(idxs)
    print('idx: ', idx_ten.shape)
    value_ten = torch.tensor(values)
    print('values: ', value_ten.shape)

    ten = torch.sparse_coo_tensor(idx_ten, value_ten)
    print('ten: ', ten.shape)

    print('Coalescing tensor...')
    ten = ten.coalesce()

    print('Saving values to cache...')
    torch.save(idx_ten, cached_idx_path)
    torch.save(value_ten, cached_val_path)

ten_nnz = calc_sparsity(ten)
print(f'Starting sparsity: {ten_nnz}')
# ten = ten.view(ten.shape[0], ten.shape[1], ten.shape[2])

parts = []
print('Aggregating along K...')
for k in range(0, ten.shape[2], AGG_SIZE_K):
    end_idx = min(ten.shape[2], k + AGG_SIZE_K)
    base = ten.select(2, k)
    for i in range(k + 1, end_idx):
        base += ten.select(2, i)
    base = base.coalesce()
    new_vals = base.values()
    new_vals[new_vals > 1] = 1
    base = torch.sparse_coo_tensor(base.indices(), base.values(), size=base.size()).coalesce()
    # print(base.shape)
    parts.append(base)

agg_ten = torch.dstack(parts)
print('Aggregated tensor:', agg_ten.shape)
ten = agg_ten
print('Coalescing aggregated tensor...')
ten = ten.coalesce()
print('Coalesced aggregated tensor:', ten.shape)

print('Summing along i')
pi = torch.sparse.sum(ten, [1, 2]).to_dense().ravel()
print('Summing along j')
pj = torch.sparse.sum(ten, [0, 2]).to_dense().ravel()
# print('Summing along k')
pk = torch.sparse.sum(ten, [0, 1]).ravel()
print(pi.shape)
print(pj.shape)
print(pk.shape)


select_i = torch.multinomial(pi, SAMPLES_I).numpy()
select_j = torch.multinomial(pj, SAMPLES_J).numpy()
select_k = torch.multinomial(pk, SAMPLES_K).numpy()

lookup_i = Dict.empty(key_type=types.int64, value_type=types.int64)
lookup_j = Dict.empty(key_type=types.int64, value_type=types.int64)
lookup_k = Dict.empty(key_type=types.int64, value_type=types.int64)

for x in range(SAMPLES_I):
    lookup_i[select_i[x]] = x

for x in range(SAMPLES_J):
    lookup_j[select_j[x]] = x

for x in range(SAMPLES_K):
    lookup_k[select_k[x]] = x
# lookup_i = { select_i[x]: x for x in range(SAMPLES_I)}
# lookup_j = { select_j[x]: x for x in range(SAMPLES_J)}
# lookup_k = { select_k[x]: x for x in range(SAMPLES_K)}

idxs = ten.indices().numpy()
print('idxs:', idxs.shape)
vals = ten.values().numpy()

print('Doing JIT part!')

@njit
def do_op(idxs, vals, I, J, K):
    # do the following operation in Numpy to be faster
    out = []
    new_vals = []

    for n in range(idxs.shape[1]):
        # print(part.shape)
        for k in range(vals.shape[1]):
            if k not in K:
                continue
            part = idxs[:, n].ravel()
            if part[0] not in I or part[1] not in J:
                continue

            val = vals[n, k]
            if abs(val) < 1e-8:
                continue

            out.append(np.array([I[part[0]], J[part[1]], K[k]]).reshape((3, 1)))
            new_vals.append(val)
    return (out, new_vals)

print('idxs:', idxs.shape)
print('vals:', vals.shape)
new_idxs, new_vals = do_op(idxs, vals, lookup_i, lookup_j, lookup_k)
t_idxs = torch.from_numpy(np.hstack(new_idxs))
t_vals = torch.from_numpy(np.array(new_vals))

print('Creating new tensor!')
new_ten = torch.sparse_coo_tensor(t_idxs, t_vals)
print('Coalescing new tensor')
new_ten = new_ten.coalesce()

print('New ten: ', new_ten.shape)

new_nnz = calc_sparsity(new_ten)
print(f'New sparsity: {new_nnz}')

# Save as numpy array
print('Converting to dense numpy array')
np_dense = new_ten.to_dense().numpy()

dest_path = path.join(path_prefix, f'enron_compressed_{new_ten.shape[0]}x{new_ten.shape[1]}x{new_ten.shape[2]}.np')
print('Saving to:', dest_path)
np.save(dest_path, np_dense)

print('Done!')

# idxs = ten.indices()
# print(idxs.shape)
# print(pi.shape, pj.shape, pk.shape)
# pi = torch.zeros(ten.shape[0])
# for i in range(ten.shape[0]):
#     pi[i] = ten[i, :, :].sum()

# pj = torch.zeros(ten.shape[1])
# for j in range(ten.shape[1]):
#     pi[j] = ten[:, j, :].sum()

# pk = torch.zeros(ten.shape[2])
# for k in range(ten.shape[2]):
#     pi[k] = ten[:, :, k].sum()
