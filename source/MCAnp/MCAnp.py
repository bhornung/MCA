"""
A native numpy implementation of the Markov cluster algorithm.
"""

import numpy as np

from BaseMCA.BaseMCA import BaseMCA


def _add_diagonal(X, scale):
    """
    Add scale * identity matrix to a matrix
    Parameters:
        X (np.ndarray[n,n]) : 2D square input matrix
        scale ({int,float}) : the number to be added to the diagonal
    Returns:
        X (np.ndarray) : matrix with modified diagonal 
    """
    
    if not isinstance(scale, (int, float)):
        raise TypeError("Scale parameter must be int or float. Got: {0}".format(type(scale)))
    
    X = X + np.eye(X.shape[0]) * scale
    
    return X


def _collect_clusters(X):
    """
    Collects those column indices together which belong to the same cluster
    Parameters:
        X np.ndarray
    """
    
    # transpose matrix for it is easier to loop over rows
    Xt = X.T
    
    # indices of nodes
    node_idcs = np.arange(Xt.shape[0], dtype = np.int)
    
    # Each row contains the indices of nodes that belong to the same cluster.
    # As a consequence, if two rows are not equal they represent to different clusters.
    
    # select nonzero rows <-- these are the clusters
    keep_mask = np.sum(Xt, axis = 1) != 0
    nnz_idcs = np.arange(Xt.shape[0])[keep_mask]
    
    # no clusters!
    if nnz_idcs.size == 0:
        raise ValueError("Cluster matrix is empty")
    
    # cluster groups will store the column index of the of the cluster vectors
    cluster_groups = [Xt[nnz_idcs[0]]]
    
    # shortcut in case of single cluster
    if nnz_idcs.size == 1:
        return {0 : node_idcs[cluster_groups[0] != 0]}
    
    # double loop to compare rows
    for idx in nnz_idcs[1:]:
        is_found = False
        for jdx in np.arange(len(cluster_groups), dtype = np.int):
    
        # compare to known clusters
        if np.allclose(Xt[idx], cluster_groups[jdx]):
            is_found = True
            break
        # append as new cluster if node pattern is not found
        if not is_found:
            cluster_groups.append(Xt[idx])
    
    # collect indices
    clusters = {idx : np.ravel(node_idcs[cluster_group != 0])
                    for idx, cluster_group in enumerate(cluster_groups)}
    
    return clusters


def _cull(X, threshold):
    """
    Sets elements below a thershold to zero
    Parameters:
        X (np.ndarray[n_nodes, n_nodes]) : 2D array
        threshold (float) : elements below this value are set to zero.
    """

    X[X < threshold] = 0.0


def _expand(X, expand_power):
    """
    Raises a square matrix to the expand_power-th power
    Parameters:
        X (np.ndarray) : 2D square matrix
        expand_power (int) : power (number) of matrix multiplications
    Returns:
        X^{expand_power}
        diff_ (np.float) : the summed absolute difference between the exponentiated and original matrix
    """
    # only integer steps are accepted
    if not isinstance(expand_power, int):
        raise TypeError("expand_power should be of type int. Got: {0}".format(type(expand_power)))
    
#   calculate matrix power
    Xm = np.linalg.matrix_power(X, expand_power)
    
    # check for idempotency
    diff_ = np.sum(np.abs(Xm-X))
    
    return Xm, diff_


def _inflate(X, inflate_power):
    """
    Raises the matrix to inflate power elementwise
    Parameters:
        X (np.ndarray) : matrix
        inflate_power : power to raise matrix to
    Returns:
        X (np.ndarray) : the elementwise exponentiated matrix
    """
  # type check powerm in order to avoid possible broadcasting
    if not isinstance(inflate_power, (int, float)):
        raise TypeError("inflate power must be float or int. Got: {0}".format(type(inflate_power)))

    # calculate elementwise power 
    return np.power(X, inflate_power)


def _init_matrix(X, diag_scale):
    """
    Generates a transition matrix from a connectivity matrix
    Parameters:
        X (np.ndarray[n_col, n_col]) : The connectivity matrix a 2D square matrix.
        diag_scale ({int, float}) : factor to scale the diagonal with.
    Returns:`
        X (np.ndarray[n_col, n_col]) : transition matrix
    """
    
    # add self loops
    X = _add_diagonal(X, diag_scale)
    
    # normalise to unit probability
    X = _row_normalise(X)
    
    return X


    def _markov_cluster(X, expand_power, inflate_power, max_iter, tol, threshold):
    """
    Performs maximum max_iter number Markov cluster cycles.
    Parameters:
        X : matrix
        expand_power (int) : number of Markov process steps
        inflate_power {int, float} : the normalising parameter for the transition probabilities.
        max_iter (int) : maximum number of iterations. 
    """
    
    # do maximum max_iter iterations
    for idx in np.arange(max_iter):
    
        # perform one cycle of Markov iteration
        X = _inflate(X, inflate_power)
        _cull(X, threshold)
        X = _row_normalise(X)
        X, diff_ = _expand(X, expand_power)
    
        # check whether convergence reached <-- matrix is idempotent
        if diff_ < tol:
            break
    
    # remove small entries, for accurate comparison
    _cull(X, threshold)
    _row_normalise(X)
    
    return X


def _row_normalise(X):
    """
    Normalises the rows of a matrix to unit
    Parameters:
        X : matrix
    Returns:
        X : rowwise normalised matrix
    """
    _row_sum = X.sum(axis = 1)
    
    # do not divide by zeros
    keep_mask = _row_sum != 0 
    X[keep_mask,:] = X[keep_mask,:] / _row_sum[keep_mask, None]
    
    return X


class MCnumpy(BaseMCA):
  """
  Native numpy implementation of the Markov Cluster algorithm.
  """


    def __init__(self, diag_scale = 1.0, expand_power = 2, inflate_power = 2, 
                 max_iter = 10, threshold = 0.00001, tol = 0.001):

        # invoke parent initialisation
        super().__init__(diag_scale = diag_scale, 
                        expand_power = expand_power, inflate_power = inflate_power,
                        max_iter = max_iter, threshold = threshold, tol = tol)


    def fit(self, X):
        """
        Clusters a graph using Markov cluster algorithm.
        Parameters:
        X (np.ndarray[n_nodes, n_nodes]): 2D connectivity matrix
        """
    
        # create transition probability matrix
        X_ = _init_matrix(X, self.diag_scale)
    
        # perform clustering
        _markov_cluster(X_, self.expand_power, self.inflate_power, 
                        self.max_iter, self.tol, self.threshold)
    
        # assign labels
        clusters = _collect_clusters(X_)
        self._labels_ = np.zeros(X_.shape[0], dtype = np.int)
    
        for _cluster_id, _cluster_members in clusters.items():
        self._labels_[_cluster_members] = _cluster_id
    
        return self


    def fit_predict(self, X):
        """
        Fits the graph with clusters and returns the cluster labels.
        Parameters:
        X (np.ndarray[n_nodes, n_nodes]): 2D connectivity matrix
        Returns:
        self.labels_ (np.ndarray[X.shape[0]]) : index of cluster to which a node belongs to.
        """
    
        return self.fit(X).labels_