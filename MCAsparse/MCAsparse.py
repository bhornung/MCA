"""
Scipy sparse implementation of the Markov cluster algorithm.
"""

import numba
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import scipy.sparse.linalg as LA 
from sklearn.preprocessing import normalize

from BaseMCA.BaseMCA import BaseMCA


#-----------------------
def _add_diagonal(X, scale):
  """
  Adds scale * identity matrix to a matrix
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    scale ({int,float}) : the number to be added to the diagonal
  Returns:
    None
  """

  if not isinstance(scale, (int, float)):
    raise TypeError("Scale parameter must be int or float. Got: {0}".format(type(scale)))

# --- fill diagonal with values
  X.setdiag(scale)

#----------------------
@numba.jit(nopython=True)
def _colour_array_numba(idcs, iptr, labels):
      """
      Assigns labels.
      Parameters:
        idcs (np.ndarray of np.int) : csr indices
        iptr (np.ndarray of np.int) : csr row pointers
        labels (np.ndarray of np.int) : initialised label array
      """

      n_nodes = iptr.size - 1  # number of nodes
      i_label = 0              # "colour" 
      n_coloured = 0           # number of coloured nodes

  # --- loop over all rows <--> vectors <--> clusters
      for i in range(n_nodes):
  # grep a cluster
          i_start, i_end = iptr[i],  iptr[i+1]
          n_nodes_in_cluster = i_start - i_end
  # skip if it is empty
          if n_nodes_in_cluster == 0: continue

          inodes = idcs[iptr[i]:iptr[i]]
  # if already coloured skip
          if labels[inodes[0]] != -1: 
              continue
          else:
  # assign colour
              labels[idcs[i_start:i_end]] = i_label
              i_label += 1
  # keep track of coloured nodes
              n_coloured += n_nodes_in_cluster
  # shortcut if all nodes have been coloured
              if n_coloured == n_nodes:
                  return

#-----------------------
def _assign_labels(X):
    """
    Assigns a label to the nodes 
    Parameters:
      X ({scipy.sparse.csr_matrix,scipy.sparse.csc_matrix}) : transition probability matrix
    Returns:
      labels (np.ndarray of np.int) : cluster label for each node
    """

    if X.format != 'csr' and X.format != 'csc':
      raise TypeError("only 'csr' and 'csc' matrices are accepted")

  # --- create an array of uniform labels
    labels = np.full(X.shape[1], -1, dtype = np.int)
  # --- colour nodes according to clusters
    _colour_array_numba(X.indices, X.indptr, labels)

    return labels

#-----------------------
def _cull(X, threshold):
    """
    Set elements to zero if they are close to zero.
    Parameters:
       X (scipy.sparse.csr_matrix) : zee matrix
    Returns:
        None
    """
    X.data[X.data < threshold] = 0.0
    X.eliminate_zeros()
    X.sort_indices()

#-----------------------
def _expand(X, expand_power):
    """
    Raises a square matrix to the expand_power-th power
    Parameters:
        X (scipy.sparse.csr_matrix) : 2D sparse matrix
        expand_power (int) : power (number) of matrix multiplications
    Returns:
        Xm scipy sparse : exponentiated matrix
        diff_ (np.float) : the sum of elementwise absolut difference between the original and exponentiated matrix
    """

    if not isinstance(expand_power, int):
       raise ValueError("expand_power must be of type int. Got: {0}".format(type(expand_power)))

    if expand_power < 1:
        raise ValueError("expand_power must be positive")

# mild danger. We are passing the reference to the original object
    if expand_power == 1:
        diff_ = 0.0
        return X, diff_

# calculate matrix power >= 2
# keep original matrix for comparison
    Xm = X.copy()

# in place multiplication
    for idx in range(1, expand_power):
      Xm *= Xm 

# deviation from being idempotent
    diff_ = abs(Xm - X).sum()

    return Xm, diff_

#----------------------- 
def _inflate(X, inflate_power):
    """
    Raises the matrix to inflate power elementwise in place.
    Parameters:
        X (scipy.sparse.csr_matrix) : 2D sparse matrix
        inflate_power : power to raise matrix to
    Returns:
        None
    """
# type check power to avoid possible broadcasting
    if not isinstance(inflate_power, (int, float)):
        raise TypeError("inflate power must be float or int. Got: {0}".format(type(inflate_power)))

# calculate elementwise power inplace
    #X.data **= inflate_power
    X = X.power(inflate_power)

    return X

#-----------------------
def _init_matrix(X, diag_scale):
    """
    Generates a transition matrix from the connectivity matrix
    Parameters:
        X (scipy.sparse.csr_matrix) : The connectivity matrix a 2D square matrix.
        diag_scale ({int, float}) : factor to scale the diagonal with. 
    Returns:
        X (scipy.sparse.csr_matrix)
    """

# --- add self loops
    _add_diagonal(X, diag_scale)

# --- normalise outgoing probabilities to unit
    _row_normalise(X)

    return X

#---------------------
def _markov_cluster(X, expand_power, inflate_power, max_iter, tol, threshold, mat_list, save_steps):
    """
    Performs maximum max_iter number Markov cluster cycles.
    Parameters:
        X : matrix
        expand_power (int) : number of Markov process steps
        inflate_power {int, float} : the normalising parameter for the transition probabilities.
        max_iter (int) : maximum number of iterations. 
    """

    if save_steps > 0:
        mat_list.append(X)

# --- do maximum max_iter iterations
    for idx in np.arange(max_iter):

# perform one cycle of Markov iteration
        X = _inflate(X, inflate_power)

        _cull(X, threshold)
        _row_normalise(X)

        if save_steps == 2:
            mat_list.append(X)

        X, diff_ = _expand(X, expand_power)

        if save_steps > 0:
            mat_list.append(X)

        print("Iteration {0} : diff {1}".format(idx, diff_))

        if diff_ < tol:
            break

    _cull(X, threshold)
    _row_normalise(X)

    return X

#-----------------------
def _row_normalise(X):
  """
  Normalises the rows of a matrix to unit in place.
  Parameters:
    X (scipy.sparse matrix) : matrix
  Returns:
    None
  """

  normalize(X, norm = 'l1', axis = 1, copy = False)

#------------------------
class MCsparse(BaseMCA):
    """
    A scipy.sparse implementation of the Markov Cluster algorithm.
    """

#-----------------------
    def __init__(self, diag_scale = 1.0, expand_power = 2, inflate_power = 2, 
                 max_iter = 10, save_steps = 0, threshold = 0.000001, tol = 0.000001):

# --- invoke parent initialisation
      super().__init__(diag_scale = diag_scale, 
                       expand_power = expand_power, inflate_power = inflate_power,
                       max_iter = max_iter, save_steps = save_steps, threshold = threshold, tol = tol)

#------------------------
    def fit(self, X):
        """
        Clusters a graph using Markov cluster algorithm.
        Parameters:
            X (scipy.sparse.csr_matrix): 2D connectivity matrix
        """

        if not isinstance(X, sps.csr_matrix):
            raise TypeError("Bad matrix format. Only 'scipy.sparse.csr_matrix' instances are accepted")

# --- create transition probability matrix
        X_ = X.copy() * 1.0
        X_ = _init_matrix(X_, self.diag_scale)

# --- perform clustering
        X_ = _markov_cluster(X_, self.expand_power, self.inflate_power, 
                             self.max_iter, self.tol, self.threshold,
                             self.mat_list, self.save_steps)
# --- assign labels

        self._labels_ = _assign_labels(X_.tocsc())

        return self

#------------------------
    def fit_predict(self, X):
        """
        Fits the graph with clusters and returns the cluster labels.
        Parameters:
            X (scipy.sparse.csr_matrix): 2D connectivity matrix
        Returns:
            self.labels_ (np.ndarray[X.shape[0]]) : index of cluster to which a node belongs to.
        """

        return self.fit(X).labels_


