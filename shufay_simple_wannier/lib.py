import numpy

def cartesian_prod(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Args:
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:

    >>> cartesian_prod(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = numpy.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = numpy.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])

    return out.reshape(nd,-1).T
