import scipy
import numpy
from numpy.typing import NDArray

# Custom modules.
import lib

class Basis():
    """
    Class to hold properties pertaining to the plane wave basis.
    """
    def __init__(
            self, 
            recip_vecs : NDArray, 
            basis : NDArray, 
            verbose : bool = False):
        self.verbose = verbose
        self.recip_vecs = recip_vecs
        self.basis = basis
        self.nbsf = self.basis.shape[0]
        self.nmax = numpy.amax(numpy.absolute(self.basis))
        
        self.direct_vecs = numpy.linalg.solve(self.recip_vecs.T, 2*numpy.pi*numpy.eye(2))
        self.L = numpy.linalg.norm(self.direct_vecs[:, 0])
        self.area = abs(scipy.linalg.det(self.direct_vecs))
    
    # TODO: the below might be unncessary.
    def build(self):
        # Build basis.
        if self.verbose: print(f'\n# Building basis...')
        max_k = self.basis @ self.recip_vecs.T # Units: 1/Bohr
        self.maxk2 = max(map(numpy.dot, max_k, max_k))
        self.basis_index_dict = self.create_basis_index_dict()
        
        # Check.
        for i, k in enumerate(self.basis):             
            imap = self.map_basis_to_spectrum_index(k)
            assert(i == imap)
        
    def create_basis_index_dict(self):
        """
        Creates a dictionary between basis indices on the reciprocal lattice
        and their energy ordering (spectrum index).
        """
        basis_idxs = list(map(lambda b: self.map_basis_to_index(b), self.basis))
        self.max_idx = max(basis_idxs)

        # Position index - index
        # Value - spectrum index
        self.basis_index_dict = numpy.zeros(self.max_idx+1, dtype=int)

        for spectrum_idx, idx in enumerate(basis_idxs):
            self.basis_index_dict[idx] = spectrum_idx

    def map_basis_to_spectrum_index(self, basis):
        """
        Maps a plane wave in the basis to its spectrum index.
        """
        if self.basis_index_dict is None:
            self.create_basis_index_dict()

        spectrum_idx = None
        idx = self.map_basis_to_index(basis)

        if (idx is not None) and (idx <= self.max_idx):
            spectrum_idx = self.basis_index_dict[idx]

        else:
            kvec = basis @ self.recip_vecs.T
            kvec2 = kvec @ kvec.T
            #print('\nkvec @ kvec.T > maxk2!')
            #print(f'kvec @ kvec.T = {kvec @ kvec.T}')

        return spectrum_idx

    def map_basis_to_index(self, basis, verbose=False):
        """
        Maps the grid representation of a plane wave in the basis to an index.
        See https://coderwall.com/p/fzni3g/
            bidirectional-translation-between-1d-and-3d-arrays.
        """
        idx = None
        kvec = basis @ self.recip_vecs.T
        kvec2 = kvec @ kvec.T

        # Check if kvec is within bound (relative to k-point).
        if (kvec2 <= self.maxk2) or numpy.allclose(kvec2-self.maxk2, 0.):
            # Index counts along direction of 1st primitive reciprocal vector.
            shift = basis + self.nmax * numpy.array([1, 1])
            idx = numpy.rint(numpy.dot(shift, numpy.array([1, (2 * self.nmax) + 1]))).astype(int)

            if verbose > 2:
                print(f'{self.nmax}')
                print(f'basis = {basis}')
                print(f'shift = {shift}')
                print(f'idx = {idx}\n')

        return idx
