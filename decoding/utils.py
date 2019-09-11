import numpy as np

from numpy.matlib import repmat


def coef_corr(x, y, var='row'):
    # Convert vectors to arrays
    if x.ndim == 1:
        x = np.array([x])
    if y.ndim == 1:
        y = np.array([y])
    # Normalize x and y to row-var format
    if var == 'row':
        # 'rowvar=1' in np.corrcoef
        # Vertical vector --> horizontal
        if x.shape[1] == 1:
            y = y.T
        if y.shape[1] == 1:
            y = y.T
    elif var == 'col':
        # 'rowvar=0' in np.corrcoef
        # Horizontal vector --> vertical
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T
        # Convert to rowvar=1
        x = x.T
        y = y.T
    else:
        raise ValueError('Unknown var parameter specified')
    # Match size of x and y
    if x.shape[0] == 1 and y.shape[0] != 1:
        x = repmat(x, y.shape[0], 1)
    elif x.shape[0] != 1 and y.shape[0] == 1:
        y = repmat(y, x.shape[0], 1)
    # Check size of x and y
    if x.shape != y.shape:
        raise TypeError('Input matrixes size mismatch')
    # Get num variables
    nvar = x.shape[0]
    # Get correlation
    rmat = np.corrcoef(x, y, rowvar=1)
    r = np.diag(rmat[:nvar, nvar:])
    return r


def select_top(data, value, num, axis=0):
    num_elem = data.shape[axis]
    sorted_index = np.argsort(value)[::-1]
    rank = np.zeros(num_elem, dtype=np.int)
    rank[sorted_index] = np.array(range(0, num_elem))
    selected_index_bool = rank < num
    if axis == 0:
        selected_data = data[selected_index_bool, :]
        selected_index = np.array(range(0, num_elem), dtype=np.int)[selected_index_bool]
    elif axis == 1:
        selected_data = data[:, selected_index_bool]
        selected_index = np.array(range(0, num_elem), dtype=np.int)[selected_index_bool]
    else:
        raise ValueError('Invalid axis')
    return selected_data, selected_index


def add_bias(x, axis=0):
    if axis == 0:
        vlen = x.shape[1]
        y = np.concatenate((x, np.array([np.ones(vlen)])), axis=0)
    elif axis == 1:
        vlen = x.shape[0]
        y = np.concatenate((x, np.array([np.ones(vlen)]).T), axis=1)
    else:
        raise ValueError('Invalid axis')
    return y

