"""
Base class of Markov cluster classifier.
"""


class BaseMCA(object):
    """
    Base class of the Markov cluster classifier.
    Attributes:
        diag_scale (float>0) : factor for scaling the diagonal of the connectivity matrix
    
        expand_power (int) : number of Markov process steps
    
        inflate_power {int, float} : the normalising parameter for the transition probabilities. Default 2.
    
        labels_ (np.ndarray, int) : list of cluster indices.
    
        mat_list [sps.sparse.csr_matrix] : list of transition matrices over iteration
    
        max_iter (int) : maximum number of iterations. Default 10. max_iter * expand_power is is the total number of Markov steps.
    
        save_steps (int) : whether to save intermediate transition matrices. Default 0. 
            0 : do not
            1 : after each it
            2 : after each inflation and expansion
    
        threshold (float) : elements below this value will be set to zero. Default 10^{-5}.
    
        tol (float) : Terminating criterion. If the L1 variance of the nonzero elements fall below tol,
            the Markov process is terminated. Default 10^{-3}
    
    Methods:
        get_params() : returns the parameters of the estimator
        set_params() : set the parameters of the estimator
    """


    def __init__(self, diag_scale = 1.0, expand_power = 2, inflate_power = 2, 
                 max_iter = 10, save_steps = 0, threshold = 0.00001, tol = 0.001):
        """
        Initialises an instance of the Markoc cluster algorithm.
    
        Parameters:
            expand_power ({int, float}) : the number of Markoc process steps. Default 2.
            inflate_power {int, float} : the normalising parameter for the transition probabilities. Default 2.
            max_iter (int) : maximum number of iterations. Default 10. max_iter * expand_power is is the total number of Markov steps.
            threshold (float) : elements below this value will be set to zero. Default 10^{-5}.
            tol (float) : Terminating criterion. If the L1 variance of the nonzero elements fall below tol, the Markov process is terminated. Default 10^{-3}
        """

        if diag_scale < 0.0:
            raise ValueError("diag_scale must be greater than zero")
        self._diag_scale = diag_scale
    
        if not isinstance(expand_power, int):
            raise TypeError("expand_power must be of type int. Got: {0}".format(type(expand_power)))
    
        if expand_power <= 0:
            raise ValueError
        self._expand_power = expand_power
    
        if inflate_power <= 0:
            raise ValueError("inflate_power must be positive")
        self._inflate_power = inflate_power
    
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self._max_iter = max_iter
    
        if not isinstance(save_steps, int):
            raise TypeError("save_steps must be of integer type")
            
        if (save_steps < 0) or (save_steps > 2):
            raise ValueError("save_steps must be 0, 1, 2. Got: {0}".format(save_steps))
        self._save_steps = save_steps
    
        if tol < 0.0:
            raise ValueError("tol must be positive")
        self._tol = tol
    
        if threshold < 0.0:
            raise ValueError("threshold must be positive")
        self._threshold = threshold
    
        self._params = {'diag_scale' : self.diag_scale, 
                        'expand_power' : self.expand_power,
                        'inflate_power' : self.inflate_power,
                        'max_iter' : self.max_iter,
                        'save_steps' : self.save_steps,
                        'threshold' : self.threshold,
                        'tol' : self.tol}
    
        self._labels_ = None
    
        self.mat_list = []

    @property
    def diag_scale(self):
        return self._diag_scale
    
    @property
    def expand_power(self):
        return self._expand_power
    
    @property
    def inflate_power(self):
        return self._inflate_power
        
    @property
    def max_iter(self):
        return self._max_iter
    
    @property
    def save_steps(self):
        return self._save_steps
    
    @property
    def threshold(self):
        return self._threshold
    
    @property
    def tol(self):
        return self._tol
    
    @property
    def labels_(self):
        return self._labels_


    def get_params(self):
        """
        Returns a dictionary of parameters. 
        Returns:
            param_dict ({}) : dictionary of parameters
        """

        # reconstruct dict in order to protect parameters
        param_dict = dict(self._params.items())

        return param_dict


    def set_params(self, param_dict):
        """
        Sets the parameters of the estimator.
        Parameters:
            param_dict : a dictionary containing the key value pairs of the parameters.
        """
        param_names = self.get_params().keys()

        for _pname, _pval in param_dict.items():
            if _pname not in param_names:
                raise KeyError("Invalid keyword {0}".format(_pname))

            self._params[_pname] = _pval

 
    def fit(self, X):
        """
        Fits clusters to the graph using Markov cluster algorithm.
        Parameters:
            X (np.ndarray[n_nodes, n_nodes]) : 2D connectivity matrix
        """

        raise NotImplementedError("fit() is not implemented in base class")

 
    def fit_predict(self, X):
        """
        Fits clusters to the graph using Markov cluster algorithm.
        Parameters:
            X (np.ndarray[n_nodes, n_nodes]) : 2D connectivity matrix
        Returns:
            self.labels_ (np.ndarray[X.shape[0]]) : index of cluster to which a node belongs to     
        """

        raise NotImplementedError("fit_predict() is not implemented in base class")

