import numpy as np
from scipy.sparse import spdiags

class HMF():
    def __init__(self, data, weights=None, nvec=5, seed=1):
        '''data[nobs, nvar], weights with same shape'''
        self.data = data
        self.weights = weights
        self.nobs = data.shape[0]
        self.nvar = data.shape[1]
        self.nvec = nvec
        self._chi2 = list()
        R = np.random.RandomState(seed)
        self.coeff = R.normal(size=(self.nvec, self.nobs))
        self.solve()
    
    def solve(self, niter=5):
        '''Iteratively solve for A and C using weights
        D[nvar,nobs] = A[nvar,nvec] C[nvec,nobs]
        self.data = D.T    #- input data
        self.mvec = A.T    #- model vectors
        self.coeff = C.T     #- coefficients of model vectors to represent data
        '''
        D = self.data.T
        C = self.coeff
        for i in range(niter):
            #- Solve D.T[nobs,nvar] = C.T A.T
            if self.weights is None:
                A = np.linalg.solve(C.dot(C.T), C.dot(D.T)).T
            else:
                #- weighted requires solving one var at a time for all vec
                W = self.weights.T
                A = np.zeros((self.nvar, self.nvec))
                for j in range(self.nvar):
                    Wx = spdiags(W[j], [0], self.nobs, self.nobs)
                    A[j] = np.linalg.solve(C.dot(Wx.dot(C.T)), C.dot(W[j]*D[j]))

            #- Normalize vectors
            for j in range(self.nvec):
                A[:,j] /= np.linalg.norm(A[:,j])
            
            #- Solve D = A C
            if self.weights is None:
                C = np.linalg.solve(A.T.dot(A), A.T.dot(D))
            else:
                #- weighted requires solving one vec at a time
                for j in range(self.nobs):
                    Wx = spdiags(W[:,j], [0], self.nvar, self.nvar)
                    C[:,j] = np.linalg.solve(A.T.dot(Wx.dot(A)), A.T.dot(W[:,j]*D[:,j]))
            
            #- Calculate model and chi2 of the current iteration fit
            M = A.dot(C)
            if self.weights is None:
                self._chi2.append( np.sum((D-M)**2) )
            else:
                self._chi2.append( np.sum((D-M)**2 * W) )

        self.mvec = A.T
        self.coeff = C.T
        self.model = M.T
    
    @property
    def chi2(self):
        return self._chi2[-1]
    
    def pca(self):
        return pca(self.model)

def pca(data):
    u, s, v = np.linalg.svd(data, full_matrices=False)
    return v
    