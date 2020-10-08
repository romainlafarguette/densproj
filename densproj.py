# -*- coding: utf-8 -*-
"""
DensityProjection class for the macro stress-test project
Romain Lafarguette, rlafarguette@imf.org
Time-stamp: "2020-08-29 06:39:42 Romain"
"""

###############################################################################
#%% Packages
###############################################################################
import os, sys, importlib                                # System packages
import pandas as pd                                      # Dataframes 
import numpy as np                                       # Numerical tools

# Functional import
from collections import namedtuple                       # High perf containers

# Local modules import
from cqsampling import inv_transform


###############################################################################
#%% Ancillary functions
###############################################################################
def col_its(col, len_sample=1000, len_bs=1000, seed=None):
    """ 
    Apply the inverse transform sampling on a pandas df column
    index: should be quantile list
    columns: conditional quantiles
    """
    cq_d = {k:v for k,v in zip(col.index, col.values)}
    ssample = inv_transform(cq_d, len_sample=len_sample,
                            len_bs=len_bs, seed=seed)
    return(ssample)


def single_sampling(col, seed=None):
    """ 
    Sample a single element on a pandas df column
    Important to keep the conditioning in correct order
    """
    element = np.random.choice(col, 1)
    return(element)


###############################################################################
#%% DensityProjection Model
###############################################################################
class DensProj(object):
    """
    Project a density, from a set of quantiles coefficients

    Inputs
    ------
    beta_coeffs: pandas dataframe
        Matrix of quantile coefficients (index: quantiles, columns=variables)
        Size: (num quantiles, num variables)


    """
    __description = "Density Projection"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"

    # Initializer
    def __init__(self, beta_coeffs):
        self.beta_coeffs = beta_coeffs
        self.quantiles_l = sorted(self.beta_coeffs.index) 
        self.vars_l = self.beta_coeffs.columns

        # For the shapes
        self.num_quantiles = len(self.quantiles_l)
        self.num_vars = len(self.vars_l)
        
        # Unit test: make sure that the inputs are correct
        self.__densproj_unittest()
    
    # Class-methods (methods which returns a class defined below)    
    def fit(self, cond_vector):
        """ Project a density based on a conditioning vector """
        return(DensFit(self, cond_vector))
        
    # Unit tests
    def __densproj_unittest(self):
        """ Unit testing on the inputs """
        m1 = 'beta_coeffs should be packaged in a pandas DataFrame'
        assert isinstance(self.beta_coeffs, pd.DataFrame), m1

        for quantile in self.quantiles_l:
            assert (0 < quantile < 1), 'quantiles index should be in (0,1)'

        for var in self.vars_l:
            assert isinstance(var, str), 'variables should be string'

            
###############################################################################
#%% DensityProjection Fit
###############################################################################
class DensFit(object): # Projection class for the QVARFit

    """ 
    Fit a density, based on a conditioning vector or matrix

    Inputs
    ------
    X: pd.DataFrame 
       Conditioning vector: should have same variables names as beta_coeffs 
       Size: (num variables, num simulations)
       
    """

    # Import from DensProj class
    def __init__(self, DensProj, X): 
        self.__dict__.update(DensProj.__dict__) # Pass all attributes

        self.X = X        
        for x in self.vars_l:
            assert x in self.X.index, f'{x} should be in conditioning vector'

        self.X = self.X.loc[self.vars_l, :].copy() # Same order as beta coeffs
        
        # Unit test: make sure that the inputs are correct
        self.__densfit_unittest()
               

        self.num_samples = self.X.shape[1]
                
        # Compute the conditional quantiles: pay attention to the dimension
        self.cond_quant = pd.DataFrame(self.beta_coeffs.dot(self.X),
                                       index=self.quantiles_l)
                
    # Public methods
    def sample(self, len_sample=1000, len_bs=1000, seed=None):
        """ 
        Inverse transform sampling on the cond quant, with uncrossing 
        
        Parameters
        ----------
          len_sample: integer, default=1000
            Length of the sample to be returned

          len_bs=integer, default=1000
            Length of the bootstrap for quantiles uncrossing

          seed:integer, default=None
            Seed of the random numbers generator, for replicability purposes
        
        """
        # Sample for each element in the conditioning vector
        # col_its is an ancillary function defined above
        dsample = self.cond_quant.apply(col_its, axis=0, len_sample=len_sample,
                                        len_bs=len_bs, seed=seed)

        # Reduce by conserving the conditioning structure
        # sample within the column: very important for joint conditioning
        sample = dsample.apply(single_sampling, axis=0).values # Keep order

        return(sample)
            
    # Unit tests
    def __densfit_unittest(self):
        """ Unit testing on the inputs """

        assert self.X.shape[1] >=1, 'X should be dataframe, not series'
        
        mshape1 = 'Conditioning vector shape should be (K,S) where '
        mshape2 = 'K is num variables, S num samples '
        assert self.X.shape[0]==self.num_vars, mshape1 + mshape2 

        for var in self.vars_l:
            assert isinstance(var, str), 'variables should be string'

        m0 = 'Conditioning vector should be a dataframe, variables as index'  
        assert isinstance(self.X, pd.DataFrame), m0    
               
        

        
