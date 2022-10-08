"""
General-purpose miscellaneous functions.
"""
import pickle
import numpy as np

def save_pickle(obj, loc, protocol=pickle.HIGHEST_PROTOCOL):
    """Saves a pickled version of `obj` to `loc`.
    Allows for quick 1-liner to save a pickle without leaving a hanging file handle.
    Useful for Jupyter notebooks.

    Also behaves differently to pickle in that it defaults to pickle.HIGHEST_PROTOCOL
        instead of pickle.DEFAULT_PROTOCOL.

    Arguments:
        obj {Any} -- The object to be pickled.
        loc {Path|str} -- A location to save the object to.
        protocol {pickle.Protocol} -- The pickle protocol level to use.
            (default {pickle.HIGHEST_PROTOCOL})
    """
    with open(loc, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pickle(loc):
    """Loads a pickled object from `loc`.
    Very helpful in avoiding overwritting a pickle when you read using 'wb'
    instead of 'rb' at 3 AM.
    Also provides a convinient 1-liner to read a pickle without leaving an open file handle.
    If we encounter a PickleError, it will try to use pickle5.

    Arguments:
        loc {Path|str} -- A location to read the pickled object from.

    Returns:
        Any -- The pickled object.
    """
    try:
        with open(loc, 'rb') as f:
            return pickle.load(f)
    except pickle.PickleError:
        # Maybe it's a pickle5 and we use Python <= 3.8.3
        import pickle5
        with open(loc, 'rb') as f:
            return pickle5.load(f)

def sample_mat(mat, n, replace=False, return_idxs=False):
    """Sample `n` random rows from mat.
    Returns an array of the rows sampled from the matrix.

    Arguments:
        mat {np.array} -- A matrix to sample rows from.
        n {int} -- The number of rows to sample.

    Keyword Arguments:
        replace {bool} -- Whether or not to sample with replacement. (default: {False})
        return_idxs {bool} -- Whether or not to return the indexes of the selected rows. (default: {False})

    Returns:
        np.array|tuple -- Returns a matrix containing the sampled rows, or if `return_idxs` is set,
            returns a tuple containing the indexes of the selected rows and the sampled matrix.
    """
    row_idxs = np.random.choice(mat.shape[0], n, replace=replace)
    sampled = mat[row_idxs, :]
    if return_idxs:
        return (row_idxs, sampled)
    return sampled

def random_interleave_mat(A, B, seed=None):
    """Randomly interleaves the rows of two Numpy matrices together. A and B can be different sizes.

    Args:
        A (np.array): The first matrix
        B (np.array): The second matrix

    Returns:
        np.array: A Numpy array with rows taken from A and B in random order.
    """
    stacked = np.vstack((A, B))
    total_rows = stacked.shape[0]
    a_rows = A.shape[0]
    rng = np.random.default_rng(seed=seed)
    row_idxs = rng.choice(total_rows, total_rows, replace=False)
    output = stacked[row_idxs, :]
    labels = (row_idxs >= a_rows)
    return output, labels

def mask_data(ten, frac=0.1, return_full_idxs=True):
    """Randomly masks out values from a N-dimensional Numpy tensor.
    Creates a copy of the input tensor.

    Args:
        ten (np.array): The tensor to pull values from.
        frac (float, optional): The ratio of values to mask out. Defaults to 0.1.

    Returns:
        (np.array, np.array, np.array): Returns 3 values:
            1. The input array with values masked out.
            2. The values taken from the input array.
            3. The indices (when using np.flat indexing) where the values were taken.
    """
    n_el = ten.size
    n_select = round(frac * n_el)
    remove_idxs = np.random.choice(n_el, n_select, replace=False)
    out = ten.copy()
    missing_vals = ten.flat[remove_idxs]
    out.flat[remove_idxs] = 0
    if return_full_idxs:
        return (out, missing_vals, np.unravel_index(remove_idxs, ten.shape))
    return (out, missing_vals, remove_idxs)

def interleave_mat(A, B, warn_on_size_diff=True):
    """Interleaves the rows of two Numpy matrices together.
    By default, it will output a warning if the two matrices are different sizes, and truncate the remaining rows.

    Args:
        A (np.array): The first matrix
        B (np.array): The second matrix
        warn_on_size_diff (bool, optional): Whether or not to output a warning on size difference. Defaults to True.

    Returns:
        np.array: A numpy array with rows, where even rows are from A and odd rows are from B.
    """
    if A.shape[1] != B.shape[1]:
        raise Exception('A and B must have the same number of columns')
    if warn_on_size_diff and A.shape[0] != B.shape[0]:
        print('Warning: A and B have a different number of rows. Some rows will be truncated.')

    min_rows = min(A.shape[0], B.shape[0])
    n_cols = A.shape[1]
    output = np.zeros((min_rows * 2, n_cols))

    for i in range(min_rows):
        output[i*2] = A[i, :]
        output[i*2+1] = B[i, :]

    return output

def create_alt_array(n, first_val = 0, second_val = 1):
    """Creates an array filled with alternating values.
    With the default values, this would result in an array of size `n` with the following pattern:

    [0, 1, 0, 1..., 0, 1]

    Args:
        n (int): The target size of the array to generate.
        first_val (int, optional): The first value to use (on even indices). Defaults to 0.
        second_val (int, optional): The second value to use (on odd indices). Defaults to 1.

    Returns:
        np.array: A numpy array with alternating values of size `n`.
    """
    arr = np.empty((n,))
    arr[::2] = first_val
    arr[1::2] = second_val
    return arr
