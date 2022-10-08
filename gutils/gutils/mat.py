"""
Matrix-related functions and classes.
Currently only contains the MarkedMatrix class.
"""

from collections import OrderedDict
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class MarkedMatrix:
    """
    A wrapper around a 2D numpy array that includes labels for each
    segment of the matrix.

    Also includes several functions to display the data.
    """

    def __init__(self, items):
        """Initializes a MarkedMatrix.
        Can take one of two types of arguments:
            1. A `list` or `tuple` of label-array tuples.
            2. A `tuple` consisting of a
                (0) data matrix (np.array)
                    - can be obtained from a MarkedMatrix with `get_mat()`
                (1) location index (OrderedDict)

        Arguments:
            items {list|tuple} -- As described above.
        """
        # Construct a MarkedMatrix from a list/tuple of lists/tuples of labels to data.
        # TODO: support all iterable types (can't use indexing)
        if isinstance(items[0], tuple) or isinstance(items[0], list):
            self.mat = np.vstack([item[1] for item in items])
            self.loc_idx = OrderedDict()
            total = 0
            for item in items:
                rows = item[1].shape[0]
                total += rows
                self.loc_idx[total] = item[0]

        # Construct a MarkedMatrix from an existing data matrix and label index
        else:
            self.mat = items[0]
            self.loc_idx = items[1]

    def get_mat(self):
        """Returns the internal matrix.
        Identical to the `mat` property.

        Returns:
            np.array -- The internal matrix of all rows in the MarkedMatrix.
        """
        return self.mat

    def get_loc_idx(self):
        """Returns the internal location matrix mapping
        the last row number to a label.

        Returns:
            OrderedDict[int, str] -- A location index.
                If a row number is less than the key, that row belongs to the
                label equal to the current value.
        """
        return self.loc_idx

    def tsne(self, tsne_args=None, ax=None, plot=True, scatter_args={}):
        """Runs TSNE on the matrix and plots the results using seaborn
        with the labels defined at the construction of the MarkedMatrix.

        Keyword Arguments:
            tsne_args {dict} -- An optional dictionary of the arguments to be passed
                to sklearn's TSNE constructor (default {None})
            ax {matplotlib.pyplot.Axis} -- An optional Axis object used to draw the plot.
                (default: {None})
            plot {bool} -- Whether or not the function should draw the plot. (default: {True})

        Returns:
            np.array -- An M x 2 numpy array of the original matrix projected
                into 2 dimensions using TSNE.
        """
        # Forward TSNE args
        if tsne_args is not None:
            tsne = TSNE(n_components=2, **tsne_args)
        else:
            tsne = TSNE(n_components=2)
        M = tsne.fit_transform(self.mat)
        # Keeps track of the start of the current section
        last_loc = 0
        if plot:
            for loc, name in self.loc_idx.items():
                sns.scatterplot(x=M[last_loc:loc, 0], y=M[last_loc:loc, 1], label=name, ax=ax, **scatter_args)
                last_loc = loc
        return M

    def get_pieces(self, mat=None):
        """Given a matrix, returns a dictionary of labels to their corresponding submatrices.
        Returns a mapping for the MarkedMatrix if no matrix is provided.

        Keyword Arguments:
            mat {np.array} -- The input matrix (default: {None})

        Returns:
            dict[str, np.array] -- A mapping from labels to matrices.
        """
        if mat is None:
            mat = self.mat
        out = OrderedDict()
        last_loc = 0
        for loc, name in self.loc_idx.items():
            out[name] = mat[last_loc:loc]
            last_loc = loc
        return out

    def single_split_classify(self, classifier, chosen_labels=None, verbose=True, return_labels=True, **kwargs):
        """Tests the performance of a classifier on the MarkedMatrix, without performing k-fold cross-validation.

        Args:
            classifier (Model): sklearn model
            chosen_labels {set|list} -- A set/list of strings of which labels to use for classification.
              Defaults to all labels.
            verbose (bool): Whether or not to print process indicators. Defaults to True.
        """
        pieces = self.get_pieces()

        if chosen_labels is None:
            M = self.mat
            classes = set(self.loc_idx.values())
        else:
            M = np.vstack([v for k, v in pieces.items() if k in chosen_labels])
            classes = set(chosen_labels)

        class_labels = {c: i for i, c in enumerate(classes)}
        f1_average = 'binary' if len(class_labels) <= 2 else None
        class_pieces = []

        for label, val in pieces.items():
            if label not in classes:
                continue
            num_items = val.shape[0]
            class_pieces.extend([class_labels[label]] * num_items)
        y = np.array(class_pieces)

        stats = {}
        if verbose:
            print('Training model...')
        X_train, X_test, y_train, y_test = train_test_split(M, y, **kwargs)
        classifier.fit(X_train, y_train)
        if verbose:
            print('Evaluating model...')
        y_hat = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat, average=f1_average)
        if return_labels:
            return ((acc, f1), class_labels)
        return (acc, f1)

    def custom_classify(self, classifiers, chosen_labels=None, n_splits=5, verbose=True):
        """Tests the performance of a classifier on the MarkedMatrix

        Arguments:
            classifiers {dict} -- A dictionary of classifier name to model of sklearn
              compatible classifiers.
            chosen_labels {set|list} -- A set/list of strings of which labels to use for classification.
              Defaults to all labels. (default: {None})

        Keyword Arguments:
            n_splits {int} -- The number of splits to use for k-fold cross-validation (default: {5})
            verbose {bool} -- Whether or not to print progress indicators. (default: {True})

        Returns:
            {dict} -- A dictionary of model name to statistics about its accuracy.
        """
        kf = KFold(n_splits=n_splits, shuffle=True)
        pieces = self.get_pieces()

        if chosen_labels is None:
            M = self.mat
            classes = set(self.loc_idx.values())
        else:
            M = np.vstack([v for k, v in pieces.items() if k in chosen_labels])
            classes = set(chosen_labels)

        class_labels = {c: i for i, c in enumerate(classes)}
        f1_average = 'binary' if len(class_labels) <= 2 else None
        class_pieces = []

        for label, val in pieces.items():
            if label not in classes:
                continue
            num_items = val.shape[0]
            class_pieces.extend([class_labels[label]] * num_items)
        y = np.array(class_pieces)

        stats = {}
        for cnt, (train_index, test_index) in enumerate(kf.split(M)):
            if verbose:
                print(f'========== Performing k-fold validation ({cnt}) ==========')
            X_train, X_test = M[train_index], M[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for name, model in classifiers.items():
                if verbose:
                    print(f'Training {name}')
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                acc = accuracy_score(y_test, y_hat)
                f1 = f1_score(y_test, y_hat, average=f1_average)
                
                if name in stats:
                    stats[name]['acc'].append(acc)
                    stats[name]['f1'].append(f1)
                else:
                    stats[name] = {
                        'acc': [acc],
                        'f1': [f1]
                    }
                
                if verbose:
                    print(f'{name} accuracy: {acc}')
                    print(f'{name} F1 score: {f1}')
                    print('-'*15)
        return stats

    def default_classify(self, **kwargs):
        """Calls `custom_classify` with some resonable default classifiers.
        Currently defaults to:
        - RandomForest
        - XGBoost
        - KNN
        - Logistic regression
        - Linear SVM

        Keyword Arguments:
           **kwargs -- Takes same arguments as custom_classify.

        Returns:
            {dict} -- A dictionary of model name to statistics about its performance.
        """
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'XGBoost': XGBClassifier(),
            'KNN': KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=500),
            'Linear SVM': LinearSVC(dual=False),
        }
        return self.custom_classify(classifiers, **kwargs)

    def distance_histogram(self, label, ax=None, nn_only=True, distance_metric='sqeuclidean'):
        labels = set(self.loc_idx.values())
        other_labels = labels - set([label])
        pieces = self.get_pieces()

        for other_label in other_labels:
            dists = cdist(pieces[other_label], pieces[label], distance_metric)
            if nn_only:
                dist_arr = np.min(dists, axis=1).ravel()
            else:
                dist_arr = dists.ravel()
            sns.distplot(dist_arr, label=other_label, ax=ax)
            plt.title(f'Distribution plot of {"NN" if nn_only else "all"} distances to {label}')
            plt.legend()

    def row_label(self, row_num):
        for loc, name in self.loc_idx.items():
            if row_num < loc:
                return name
        return None

    @staticmethod
    def from_df(df, data_col, label_col, labels=None):
        if labels is None:
            labels = set(df[label_col])
        mapping = []

        for label in labels:
            sub_df = df[df[label_col] == label]
            mapping.append((label, np.vstack(list(sub_df[data_col]))))
        return MarkedMatrix(mapping)
