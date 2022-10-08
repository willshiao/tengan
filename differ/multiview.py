import math
from re import split
import scipy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import mode
from karateclub.graph_embedding.ldp import LDP
from karateclub.node_embedding.neighbourhood import DeepWalk
import torch
from gutils.general import random_interleave_mat, create_alt_array
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import logging

# Set logging to INFO to see RESCAL information
logging.basicConfig(level=logging.INFO)

class DeepWalkWrapper():
    def __init__(self, agg_method='mean', **kwargs):
        self.kwargs = kwargs
        self.embedder = DeepWalk(**kwargs, workers=20)
        self.agg_method = agg_method

    def fit(self, graphs):
        embedding_list = []
        for graph in graphs:
            self.embedder.fit(graph)
            node_embeddings = self.embedder.get_embedding()
            if self.agg_method == 'mean':
                graph_embedding = node_embeddings.mean(axis=0)
            elif self.agg_method == 'max':
                graph_embedding = node_embeddings.max(axis=0)
            embedding_list.append(graph_embedding)
        self.embeddings = np.vstack(embedding_list)

    def get_embedding(self):
        return self.embeddings

class MultiviewDiffer:
    def __init__(self, real_gs, gen_gs, model=None, split_first=False, even_out=False, embedding_model=None, verbose=True, use_shared_model=True, use_mv_embedding=False, use_ensemble_model=True):
        self.real_gs = real_gs
        self.gen_gs = gen_gs
        # self.real_train, self.real_test, self.gen_train, self.gen_test = train_test_split(real_gs, gen_gs)
        self.verbose = verbose
        self.fitted = False
        self.use_mv_embedding = use_mv_embedding
        self.use_shared_model = use_shared_model
        self.n_views = len(self.real_gs[0])
        self.split_first = split_first
        self.use_ensemble_model = use_ensemble_model
        self.rng = np.random.default_rng()
        self.even_out = even_out

        if model is None:
            print('Warning: no model specified, defaulting to SVC model.')
            if use_ensemble_model:
                self.models = [SVC() for _ in range(self.n_views)]
            else:
                self.model = SVC()

        if embedding_model is None:
            print('Warning: no embedding model specified, defaulting to LDP model.')
            self.embedding_model = LDP()
        else:
            self.embedding_model = embedding_model

        if self.use_ensemble_model:
            self.ensemble_embed()
        else:
            self.embed()

    def vprint(self, *args):
        if (self.verbose):
            print(*args)

    def embed(self):
        if self.use_mv_embedding:
            self.vprint('Training embedding model for real samples')
            self.embedding_model.fit(self.real_gs)
            self.vprint('Done training embedding model for real samples')
            self.real_embeddings = self.embedding_model.get_embedding()

            self.vprint('Training embedding model for generated samples')
            self.embedding_model.fit(self.gen_gs)
            self.vprint('Done training embedding model for generated samples')
            self.gen_embeddings = self.embedding_model.get_embedding()
        else: # use graph embeddings, and concat to form multiview embedding
            slices = len(self.real_gs[0])
            if slices != len(self.gen_gs[0]):
                raise Exception(f'Generated graphs have different # of modes ({slices} vs {len(self.gen_gs[0])})')

            all_real_slices = []
            for g in self.real_gs:
                for view in g:
                    relabeled = nx.relabel_nodes(view, {x: i for i, x in enumerate(sorted(view))})
                    all_real_slices.append(relabeled)

            all_gen_slices = []
            for g in self.gen_gs:
                for view in g:
                    relabeled = nx.relabel_nodes(view, {x: i for i, x in enumerate(sorted(view))})
                    all_gen_slices.append(relabeled)

            if self.use_shared_model:
                n_real = len(all_real_slices)
                self.vprint('Training embedding model')
                self.embedding_model.fit(all_real_slices + all_gen_slices)
                all_embeddings = self.embedding_model.get_embedding()
                all_real_embeddings = all_embeddings[:n_real, :]
                all_gen_embeddings = all_embeddings[n_real:, :]
                self.vprint('Done training embedding model')
            else:
                self.vprint('Training embedding model for real samples')
                self.embedding_model.fit(all_real_slices)
                self.vprint('Done training embedding model for real samples')
                all_real_embeddings = self.embedding_model.get_embedding()

                self.vprint('Training embedding model for generated samples')
                self.embedding_model.fit(all_gen_slices)
                self.vprint('Done training embedding model for generated samples')
                all_gen_embeddings = self.embedding_model.get_embedding()

            self.real_embeddings = all_real_embeddings.reshape((all_real_embeddings.shape[0] // slices, all_real_embeddings.shape[1] * slices))
            self.gen_embeddings = all_gen_embeddings.reshape((all_gen_embeddings.shape[0] // slices, all_gen_embeddings.shape[1] * slices))


        self.real_train_embeddings, self.real_test_embeddings = train_test_split(self.real_embeddings)
        self.gen_train_embeddings, self.gen_test_embeddings = train_test_split(self.gen_embeddings)

        self.train_embeddings = random_interleave_mat(self.real_train_embeddings, self.gen_train_embeddings)
        self.train_labels = create_alt_array(self.train_embeddings.shape[0])
        self.test_embeddings = random_interleave_mat(self.real_test_embeddings, self.gen_test_embeddings)
        self.test_labels = create_alt_array(self.test_embeddings.shape[0])
        # TODO: maybe shuffle to see if that makes a difference?

    def ensemble_embed(self):
        slices = len(self.real_gs[0])
        if slices != len(self.gen_gs[0]):
            raise Exception(f'Generated graphs have different # of modes ({slices} vs {len(self.gen_gs[0])})')

        all_real_slices = [[] for _ in range(self.n_views)]
        for g in self.real_gs:
            for i, view in enumerate(g):
                relabeled = nx.relabel_nodes(view, {x: i for i, x in enumerate(sorted(view))})
                all_real_slices[i].append(relabeled)

        all_gen_slices = [[] for _ in range(self.n_views)]
        for g in self.gen_gs:
            for i, view in enumerate(g):
                relabeled = nx.relabel_nodes(view, {x: i for i, x in enumerate(sorted(view))})
                all_gen_slices[i].append(relabeled)

        self.real_embeddings = []
        self.gen_embeddings = []
        self.real_train_embeddings = []
        self.real_test_embeddings = []
        self.gen_train_embeddings = []
        self.gen_test_embeddings = []
        self.train_embeddings = []
        self.test_embeddings = []
        self.train_labels = []
        self.test_labels = []

        seeds = [self.rng.integers(100000) for _ in range(4)]

        for v in range(self.n_views):
            self.vprint(f'Split first: {self.split_first}')

            if self.split_first:
                if self.even_out:
                    min_sz = min(len(all_gen_slices[v]), len(all_real_slices[v]))
                    all_real_slices[v] = all_real_slices[v][:min_sz]
                    all_gen_slices[v] = all_gen_slices[v][:min_sz]
                train_real_slices, test_real_slices = train_test_split(all_real_slices[v], random_state=seeds[0])
                train_gen_slices, test_gen_slices = train_test_split(all_gen_slices[v], random_state=seeds[1])
                if self.use_shared_model:
                    self.vprint(f'Training embedding model for view {v}')
                    n_rows = len(train_real_slices)
                    all_slices = train_real_slices + train_gen_slices
                    self.embedding_model.fit(all_slices)
                    self.vprint(f'Done training embedding model for view {v}')
                    slice_embeddings = self.embedding_model.get_embedding()
                    self.real_train_embeddings.append(slice_embeddings[:n_rows, :])
                    self.gen_train_embeddings.append(slice_embeddings[n_rows:, :])
                else:
                    self.vprint(f'Training embedding model for real samples for view {v}')
                    self.embedding_model.fit(train_real_slices)
                    self.vprint(f'Done training embedding model for real samples for view {v}')
                    self.real_train_embeddings.append(self.embedding_model.get_embedding())

                    # maybe we should use the same embedding model? not sure if we should retrain when running on generated samples...
                    self.vprint(f'Training embedding model for generated samples for view {v}')
                    self.embedding_model.fit(train_gen_slices)
                    self.vprint(f'Done training embedding model for generated samples for view {v}')
                    self.gen_train_embeddings.append(self.embedding_model.get_embedding())

                # should use a for loop, but i'm tired...
                if self.use_shared_model:
                    self.vprint(f'Training embedding model for view {v}')
                    n_rows = len(test_real_slices)
                    all_slices = test_real_slices + test_gen_slices
                    self.embedding_model.fit(all_slices)
                    self.vprint(f'Done training embedding model for view {v}')
                    slice_embeddings = self.embedding_model.get_embedding()
                    self.real_test_embeddings.append(slice_embeddings[:n_rows, :])
                    self.gen_test_embeddings.append(slice_embeddings[n_rows:, :])
                else:
                    self.vprint(f'Training embedding model for real samples for view {v}')
                    self.embedding_model.fit(test_real_slices)
                    self.vprint(f'Done training embedding model for real samples for view {v}')
                    self.real_test_embeddings.append(self.embedding_model.get_embedding())

                    # maybe we should use the same embedding model? not sure if we should retrain when running on generated samples...
                    self.vprint(f'Training embedding model for generated samples for view {v}')
                    self.embedding_model.fit(test_gen_slices)
                    self.vprint(f'Done training embedding model for generated samples for view {v}')
                    self.gen_test_embeddings.append(self.embedding_model.get_embedding())

                # self.vprint(f'real_train_embeddings sz: {len(self.real_train_embeddings)}')
                # self.vprint(f'gen_train_embeddings sz: {len(self.gen_train_embeddings)}')
                train_embeddings, train_labels = random_interleave_mat(self.real_train_embeddings[v], self.gen_train_embeddings[v], seed=seeds[2])
                test_embeddings, test_labels = random_interleave_mat(self.real_test_embeddings[v], self.gen_test_embeddings[v], seed=seeds[3])
                self.train_embeddings.append(train_embeddings)
                self.train_labels.append(train_labels)
                self.test_embeddings.append(test_embeddings)
                self.test_labels.append(test_labels)
            else:
                if self.even_out:
                    min_sz = min(len(all_gen_slices[v]), len(all_real_slices[v]))
                    all_real_slices[v] = all_real_slices[v][:min_sz]
                    all_gen_slices[v] = all_gen_slices[v][:min_sz]
                if self.use_shared_model:
                    self.vprint(f'Training embedding model for view {v}')
                    n_rows = len(all_real_slices[v])
                    all_slices = all_real_slices[v] + all_gen_slices[v]
                    self.embedding_model.fit(all_slices)
                    self.vprint(f'Done training embedding model for view {v}')
                    slice_embeddings = self.embedding_model.get_embedding()
                    self.real_embeddings.append(slice_embeddings[:n_rows, :])
                    self.gen_embeddings.append(slice_embeddings[n_rows:, :])
                else:
                    self.vprint(f'Training embedding model for real samples for view {v}')
                    self.embedding_model.fit(all_real_slices[v])
                    self.vprint(f'Done training embedding model for real samples for view {v}')
                    self.real_embeddings.append(self.embedding_model.get_embedding())

                    # maybe we should use the same embedding model? not sure if we should retrain when running on generated samples...
                    self.vprint(f'Training embedding model for generated samples for view {v}')
                    self.embedding_model.fit(all_gen_slices[v])
                    self.vprint(f'Done training embedding model for generated samples for view {v}')
                    self.gen_embeddings.append(self.embedding_model.get_embedding())

                a, b = train_test_split(self.real_embeddings[v], random_state=seeds[0])
                self.real_train_embeddings.append(a)
                self.real_test_embeddings.append(b)
                c, d = train_test_split(self.gen_embeddings[v], random_state=seeds[1])
                self.gen_train_embeddings.append(c)
                self.gen_test_embeddings.append(d)

                # self.vprint(f'real_train_embeddings sz: {len(self.real_train_embeddings)}')
                # self.vprint(f'gen_train_embeddings sz: {len(self.gen_train_embeddings)}')
                train_embeddings, train_labels = random_interleave_mat(self.real_train_embeddings[v], self.gen_train_embeddings[v])#, seed=seeds[2])
                test_embeddings, test_labels = random_interleave_mat(self.real_test_embeddings[v], self.gen_test_embeddings[v], seed=seeds[3])
                self.train_embeddings.append(train_embeddings)
                self.train_labels.append(train_labels)
                self.test_embeddings.append(test_embeddings)
                self.test_labels.append(test_labels)


    def fit(self):
        if self.fitted:
            print('Warning: model is already fitted')
        if self.use_ensemble_model:
            self.ensemble_fit()
        else:
            self.model.fit(self.train_embeddings, self.train_labels)
        self.fitted = True

    def ensemble_fit(self):
        for i, mdl in enumerate(self.models):
            mdl.fit(self.train_embeddings[i], self.train_labels[i])

    def eval(self):
        if not self.fitted:
            self.fit()
        if self.use_ensemble_model:
            print(self.test_labels)
            # Sanity check:
            for i in range(len(self.test_labels)):
                if not np.array_equal(self.test_labels[0], self.test_labels[i]):
                    print('NOT EQUAL!!!: ', self.test_labels[0], self.test_labels[i])
                    raise AssertionError()
            y_pred = self.ensemble_pred()
            test_labels = self.test_labels[0]
        else:
            y_pred = self.model.predict(self.test_embeddings)
            test_labels = self.test_labels

        print('test labels sz:', (test_labels.shape))
        return {
            'f1': f1_score(test_labels, y_pred),
            'accuracy': accuracy_score(test_labels, y_pred)
        }

    def ensemble_pred(self):
        preds = []
        for i, mdl in enumerate(self.models):
            preds.append(mdl.predict(self.test_embeddings[i]))
        pred_ten = np.stack(preds, axis=0)
        overall_pred = mode(pred_ten, axis=0)[0].ravel()
        self.vprint(f'test_embeddings sz: {self.test_embeddings[0].shape}')
        self.vprint(f'Before mode: {pred_ten.shape}, after mode: {overall_pred.shape}')
        return overall_pred

    def get_model(self):
        return self.model


import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from tensorly import set_backend
set_backend('pytorch')

class TensorDiffer:
    def __init__(self, real_gs, gen_gs, verbose=True, use_emd=False):
        self.real_gs = real_gs
        self.gen_gs = gen_gs
        self.gen_tens = []
        for idx, x in enumerate(self.gen_gs):
            layers = []
            for y in x:
                converted = nx.to_numpy_array(y)
                # TODO: change 50 to general size
                layers.append(np.pad(converted, pad_width=(50 - converted.shape[0], 0)))
            # print(idx, [y.shape for y in layers])
            self.gen_tens.append(np.stack(layers, axis=0))
        self.real_tens = [np.stack([nx.to_numpy_array(y) for y in x], axis=0) for x in self.real_gs]
        # print('real_gs: ', real_gs)
        # print(len(real_gs))
        # print(len(real_gs[0]))

    def eval(self, max_rank=20):
        self.real_decomps = []
        self.gen_decomps = []

        self.real_idx = np.random.choice((len(self.real_tens)))
        self.gen_idx = np.random.choice((len(self.gen_tens)))
        for r in range(1, max_rank + 1):
            print(f'Calculating rank {r} decomposition')
            real_ten = torch.Tensor(self.real_tens[self.real_idx])
            gen_ten = torch.Tensor(self.gen_tens[self.gen_idx])
            print('real_ten.shape: ', real_ten.shape)
            self.real_decomps.append(parafac(real_ten, r, return_errors=True, init='random'))
            self.gen_decomps.append(parafac(gen_ten, r, return_errors=True, init='random'))
        return (self.real_decomps, self.gen_decomps)

    def pairwise_sampled_eval(self, n_samples=50, max_rank=40):
        n_real = len(self.real_tens)
        n_gen = len(self.gen_tens)

        self.real_idxs = np.random.choice(n_real, replace=True, size=n_samples)
        self.gen_idxs = np.random.choice(n_gen, replace=True, size=n_samples)

        self.real_results = np.zeros((n_samples, max_rank))
        self.gen_results = np.zeros((n_samples, max_rank))

        for i in tqdm(range(n_samples), desc='Calculating decompositions'):
            for r in range(max_rank):
                actual_rank = r + 1
                # print(f'Calculating rank {actual_rank} decomposition')
                real_ten = torch.Tensor(self.real_tens[self.real_idxs[i]])
                gen_ten = torch.Tensor(self.gen_tens[self.gen_idxs[i]])

                try:
                    real_decomp = parafac(real_ten, actual_rank, return_errors=True, init='random')
                    gen_decomp = parafac(gen_ten, actual_rank, return_errors=True, init='random')
                    self.real_results[i, r] = real_decomp[1][-1]
                    self.gen_results[i, r] = gen_decomp[1][-1]
                except RuntimeError as e:
                    print('GOT ERROR RUNNING PARAFAC:', e)
                    self.real_results[i, r] = np.NaN
                    self.gen_results[i, r] = np.NaN

                # self.real_decomps.append(parafac(real_ten, r, return_errors=True))
                # self.gen_decomps.append(parafac(gen_ten, r, return_errors=True))

        # calculate wasserstein distance for all sampled pairs
        dists = []
        self_dists = []

        for i in tqdm(range(self.real_idxs.shape[0]), desc='Calculating real-pred wasserstein distances'):
            for j in range(self.gen_idxs.shape[0]):
                real_row = self.real_results[i, :]
                real_row = real_row[np.isfinite(real_row)]
                gen_row = self.gen_results[j, :]
                gen_row = gen_row[np.isfinite(gen_row)]
                # real_idx = self.real_idxs[i]
                # gen_idx = self.gen_idxs[j]

                try:
                    dists.append(wasserstein_distance(real_row, gen_row))
                except ValueError as e:
                    print('Got error (likely empty distribution):', e)
                    continue
        for i in tqdm(range(self.real_idxs.shape[0]), desc='Calculating real-real wasserstein distances'):
            for j in range(self.real_idxs.shape[0]):
                if i == j:
                    continue
                real_row = self.real_results[i, :]
                real_row = real_row[np.isfinite(real_row)]
                real_row2 = self.real_results[j, :]
                real_row2 = real_row2[np.isfinite(real_row2)]

                try:
                    self_dists.append(wasserstein_distance(real_row, real_row2))
                except ValueError as e:
                    print('Got error (likely empty distribution):', e)
                    continue
        return dists, self_dists


    def get_losses(self):
        return [float(x[1][-1]) for x in self.real_decomps], [float(x[1][-1]) for x in self.gen_decomps]

    def get_idxs(self):
        return self.real_idx, self.gen_idx

    def plot(self, show=True, save_path=None):
        self.real_losses = [x[1][-1] for x in self.real_decomps]
        self.gen_losses = [x[1][-1] for x in self.gen_decomps]
        print(self.real_losses[0].shape)
        plt.plot(self.real_losses, label=f'Real losses (idx {self.real_idx})')
        plt.plot(self.gen_losses, label=f'Gen losses (idx {self.gen_idx})')
        plt.legend()
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)


class RescalDiffer:
    def __init__(self, real_gs, gen_gs, verbose=True, use_emd=False):
        self.real_gs = real_gs
        self.gen_gs = gen_gs
        self.gen_tens = []
        for idx, x in enumerate(self.gen_gs):
            layers = []
            for y in x:
                converted = nx.to_numpy_array(y)
                # TODO: change 50 to general size
                layers.append(np.pad(converted, pad_width=(50 - converted.shape[0], 0)))
            # print(idx, [y.shape for y in layers])
            self.gen_tens.append(np.stack(layers, axis=0))
        self.real_tens = [np.stack([nx.to_numpy_array(y) for y in x], axis=0) for x in self.real_gs]
        # print('real_gs: ', real_gs)
        # print(len(real_gs))
        # print(len(real_gs[0]))

    def eval(self, max_rank=20):
        self.real_decomps = []
        self.gen_decomps = []

        self.real_idx = np.random.choice((len(self.real_tens)))
        self.gen_idx = np.random.choice((len(self.gen_tens)))
        for r in range(1, max_rank + 1):
            print(f'Calculating rank {r} decomposition')
            real_ten = torch.Tensor(self.real_tens[self.real_idx])
            gen_ten = torch.Tensor(self.gen_tens[self.gen_idx])
            print('real_ten.shape: ', real_ten.shape)
            self.real_decomps.append(parafac(real_ten, r, return_errors=True, init='random'))
            self.gen_decomps.append(parafac(gen_ten, r, return_errors=True, init='random'))
        return (self.real_decomps, self.gen_decomps)

    def pairwise_sampled_eval(self, n_samples=50, max_rank=40):
        n_real = len(self.real_tens)
        n_gen = len(self.gen_tens)

        self.real_idxs = np.random.choice(n_real, replace=True, size=n_samples)
        self.gen_idxs = np.random.choice(n_gen, replace=True, size=n_samples)

        self.real_results = np.zeros((n_samples, max_rank))
        self.gen_results = np.zeros((n_samples, max_rank))

        for i in tqdm(range(n_samples), desc='Calculating decompositions'):
            for r in range(max_rank):
                actual_rank = r + 1
                # print(f'Calculating rank {actual_rank} decomposition')
                real_ten = torch.Tensor(self.real_tens[self.real_idxs[i]])
                gen_ten = torch.Tensor(self.gen_tens[self.gen_idxs[i]])

                try:
                    real_core, real_fact = partial_tucker(real_ten, modes=[0, 1], rank=actual_rank, init='random')
                    gen_core, gen_fact = partial_tucker(gen_ten, modes=[0, 1], rank=actual_rank, init='random')
                    real_norm = tl.norm(real_ten, 2)
                    real_rec_err = math.sqrt(abs(real_norm**2 - tl.norm(real_core, 2)**2)) / real_norm
                    gen_norm = tl.norm(gen_ten, 2)
                    gen_rec_err = math.sqrt(abs(gen_norm**2 - tl.norm(gen_core, 2)**2)) / gen_norm

                    self.real_results[i, r] = real_rec_err
                    self.gen_results[i, r] = gen_rec_err
                except RuntimeError as e:
                    print('GOT ERROR RUNNING PARAFAC:', e)
                    self.real_results[i, r] = np.NaN
                    self.gen_results[i, r] = np.NaN

                # self.real_decomps.append(parafac(real_ten, r, return_errors=True))
                # self.gen_decomps.append(parafac(gen_ten, r, return_errors=True))

        # calculate wasserstein distance for all sampled pairs
        dists = []
        self_dists = []

        for i in tqdm(range(self.real_idxs.shape[0]), desc='Calculating real-pred wasserstein distances'):
            for j in range(self.gen_idxs.shape[0]):
                real_row = self.real_results[i, :]
                real_row = real_row[np.isfinite(real_row)]
                gen_row = self.gen_results[j, :]
                gen_row = gen_row[np.isfinite(gen_row)]
                # real_idx = self.real_idxs[i]
                # gen_idx = self.gen_idxs[j]

                try:
                    dists.append(wasserstein_distance(real_row, gen_row))
                except ValueError as e:
                    print('Got error (likely empty distribution):', e)
                    continue
        for i in tqdm(range(self.real_idxs.shape[0]), desc='Calculating real-real wasserstein distances'):
            for j in range(self.real_idxs.shape[0]):
                if i == j:
                    continue
                real_row = self.real_results[i, :]
                real_row = real_row[np.isfinite(real_row)]
                real_row2 = self.real_results[j, :]
                real_row2 = real_row2[np.isfinite(real_row2)]

                try:
                    self_dists.append(wasserstein_distance(real_row, real_row2))
                except ValueError as e:
                    print('Got error (likely empty distribution):', e)
                    continue
        return dists, self_dists


    def get_losses(self):
        return [float(x[1][-1]) for x in self.real_decomps], [float(x[1][-1]) for x in self.gen_decomps]

    def get_idxs(self):
        return self.real_idx, self.gen_idx

    def plot(self, show=True, save_path=None):
        self.real_losses = [x[1][-1] for x in self.real_decomps]
        self.gen_losses = [x[1][-1] for x in self.gen_decomps]
        print(self.real_losses[0].shape)
        plt.plot(self.real_losses, label=f'Real losses (idx {self.real_idx})')
        plt.plot(self.gen_losses, label=f'Gen losses (idx {self.gen_idx})')
        plt.legend()
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)


class RescalDiffer2:
    def __init__(self, real_gs, gen_gs, verbose=True):
        self.real_gs = real_gs
        self.gen_gs = gen_gs
        self.gen_tens = []
        for idx, x in enumerate(self.gen_gs):
            layers = []
            for y in x:
                converted = nx.to_numpy_array(y)
                converted = np.pad(converted, pad_width=(50 - converted.shape[0], 0))
                # TODO: change 50 to general size
                layers.append(scipy.sparse.lil_matrix(converted))
            # print(idx, [y.shape for y in layers])
            self.gen_tens.append(layers)
        self.real_tens = [[nx.to_scipy_sparse_matrix(y, format='lil') for y in x] for x in self.real_gs]
        # print('real_gs: ', real_gs)
        # print(len(real_gs))
        # print(len(real_gs[0]))

    def pairwise_sampled_eval(self, n_samples=50, max_rank=40):
        from rescal import rescal_als

        n_real = len(self.real_tens)
        n_gen = len(self.gen_tens)

        self.real_idxs = np.random.choice(n_real, replace=True, size=n_samples)
        self.gen_idxs = np.random.choice(n_gen, replace=True, size=n_samples)

        self.real_results = np.empty((n_samples, max_rank))
        self.gen_results = np.empty((n_samples, max_rank))
        self.gen_results[:] = np.NaN
        self.real_results[:] = np.NaN

        for i in tqdm(range(n_samples), desc='Calculating decompositions'):
            for r in range(max_rank):
                actual_rank = r + 1
                # print(f'Calculating rank {actual_rank} decomposition')
                real_ten = self.real_tens[self.real_idxs[i]]
                gen_ten = self.gen_tens[self.gen_idxs[i]]

                try:
                    real_decomp = rescal_als(real_ten, actual_rank, compute_fit=True)
                    gen_decomp = rescal_als(gen_ten, actual_rank, compute_fit=True)
                    # real_decomp = parafac(real_ten, actual_rank, return_errors=True, init='random')
                    # gen_decomp = parafac(gen_ten, actual_rank, return_errors=True, init='random')
                    print('got fit: ', real_decomp[2], gen_decomp[2])
                    self.real_results[i, r] = real_decomp[2]
                    self.gen_results[i, r] = gen_decomp[2]
                except RuntimeError as e:
                    print('GOT ERROR RUNNING PARAFAC:', e)
                    self.real_results[i, r] = np.NaN
                    self.gen_results[i, r] = np.NaN
                except np.linalg.LinAlgError as le:
                    print('GOT ERROR RUNNING RESCAL:', le)
                    self.real_results[i, r] = np.NaN
                    self.gen_results[i, r] = np.NaN
                except TypeError as te:
                    print('Got TypeError runnign RESCAL:', te)
                    break


                # self.real_decomps.append(parafac(real_ten, r, return_errors=True))
                # self.gen_decomps.append(parafac(gen_ten, r, return_errors=True))

        # calculate wasserstein distance for all sampled pairs
        dists = []
        self_dists = []

        for i in tqdm(range(self.real_idxs.shape[0]), desc='Calculating real-pred wasserstein distances'):
            for j in range(self.gen_idxs.shape[0]):
                real_row = self.real_results[i, :]
                real_row = real_row[np.isfinite(real_row)]
                gen_row = self.gen_results[j, :]
                gen_row = gen_row[np.isfinite(gen_row)]
                # real_idx = self.real_idxs[i]
                # gen_idx = self.gen_idxs[j]
                dists.append(wasserstein_distance(real_row, gen_row))

        for i in tqdm(range(self.real_idxs.shape[0]), desc='Calculating real-real wasserstein distances'):
            for j in range(self.real_idxs.shape[0]):
                if i == j:
                    continue
                real_row = self.real_results[i, :]
                real_row = real_row[np.isfinite(real_row)]
                real_row2 = self.real_results[j, :]
                real_row2 = real_row2[np.isfinite(real_row2)]

                try:
                    self_dists.append(wasserstein_distance(real_row, real_row2))
                except ValueError as e:
                    print('Got error (likely empty distribution):', e)
                    continue
        return dists, self_dists


    def get_losses(self):
        return [float(x[1][-1]) for x in self.real_decomps], [float(x[1][-1]) for x in self.gen_decomps]

    def get_idxs(self):
        return self.real_idx, self.gen_idx

    def plot(self, show=True, save_path=None):
        self.real_losses = [x[1][-1] for x in self.real_decomps]
        self.gen_losses = [x[1][-1] for x in self.gen_decomps]
        print(self.real_losses[0].shape)
        plt.plot(self.real_losses, label=f'Real losses (idx {self.real_idx})')
        plt.plot(self.gen_losses, label=f'Gen losses (idx {self.gen_idx})')
        plt.legend()
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
