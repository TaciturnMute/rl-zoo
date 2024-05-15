def CompleteShape(data: np.ndarray):

    batch_size = data.shape[0]  # first dim of data is batch_size by default
    if len(data.shape) == 2:    # each data in the batch is a vector or a scalar
        if data.shape[1] > 1:     # vector
            dim1 = data.shape[1]
            data = data.reshape(batch_size, 1, dim1)
            return data
        else:
            return data           # scalar and has complete shape
    elif len(data.shape) == 1:   # each data in this batch is scalar but has no complete shape
        return data.reshape(batch_size, 1)
    else: # each data in the batch is at least two or higher dimensionality, do not need to process
        return data