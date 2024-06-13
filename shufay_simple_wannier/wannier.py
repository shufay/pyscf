import scipy
import numpy
from numba import njit
from numpy.typing import NDArray

# custom modules.
import lib
import utils
from lo_base import KLocalizedOrbitals
from planewaves import Basis
from kpoints import KPoints
import gauge_fixing as gf
    
@njit(fastmath=True)
def _get_dm_grid(weight, coeff_grid):
    return (coeff_grid @ coeff_grid.T.conj()) * weight

@njit(fastmath=True)
def _get_ovlp_grid(weight, coeff_grid):
    return (coeff_grid.T.conj() @ coeff_grid) * weight

@njit(fastmath=True)
def _get_rho_grid(weight, coeff_grid):
    return (coeff_grid.conj() * coeff_grid) * weight

@njit(fastmath=True)
def _get_X(kpts, grid, area):
    nkpt = kpts.shape[0]
    ngrid = grid.shape[0]
    X = numpy.zeros((ngrid, nkpt), dtype=numpy.complex128)

    for ir, r in enumerate(grid):
        for ik, k in enumerate(kpts):
            X[ir, ik] = numpy.exp(1.j * (k @ r))

    return X / numpy.sqrt(area)

@njit(fastmath=True)
def _get_Y(basis, recip_vecs, grid):
    nbsf = basis.shape[0]
    ngrid = grid.shape[0]
    Y = numpy.zeros((ngrid, nbsf), dtype=numpy.complex128)

    for ir, r in enumerate(grid):
        for iG, G in enumerate(basis):
            Gvec = G @ recip_vecs.T
            Y[ir, iG] = numpy.exp(1.j * (Gvec @ r))

    return Y


class Wannier(KLocalizedOrbitals):
    """
    Class to obtain localized orbitals at lattice sites via simple Wannierization
    of Bloch wavefunctions.
    """
    def __init__(
            self, 
            mo_coeff: NDArray, 
            basis_obj: Basis, 
            kpts_obj: KPoints,
            verbose: bool = False):
        super().__init__(mo_coeff, basis_obj, kpts_obj, verbose)

        # To compute.
        self.R_shift = None
        self.sites = None

    def get_bloch_coeff_grid(self, grid=None, weight=None):
        if grid is None: grid, weight = self.get_grid()
        ngrid = grid.shape[0]
        bloch_coeff_grid = numpy.zeros((ngrid, self.nsite), dtype=numpy.complex128)

        for ik, k in enumerate(self.kpts):
            kvecs = self.basis @ self.recip_vecs_u.T + k
            ao = utils.eval_ao(kvecs, grid, self.area_u)
            bloch_coeff_grid[:, ik] = ao @ self.mo_coeff[:, ik]

        # Normalize.
        ovlp_grid = self.get_ovlp_grid(weight, bloch_coeff_grid)
        ovlp_diag = numpy.diag(ovlp_grid)
        bloch_coeff_grid /= numpy.sqrt(ovlp_diag)

        # Check overlaps.
        ovlp_grid = self.get_ovlp_grid(weight, bloch_coeff_grid)
        ovlp_diag = numpy.diag(ovlp_grid)
        ovlp_offdiag = ovlp_grid - numpy.diag(ovlp_diag)
        numpy.testing.assert_allclose(ovlp_diag, 1., atol=1e-12)
        numpy.testing.assert_allclose(ovlp_offdiag, 0., atol=1e-12)
   
        return bloch_coeff_grid

    def get_R_shift(self, grid, weight, coeff_grid):
        # !! Needs a sufficiently dense grid!
        rho_grid = self.get_rho_grid(weight, coeff_grid)
        peaks = []

        for i, rho_i in enumerate(rho_grid.T):
            peak_idx = numpy.argmax(rho_i)
            R = grid[peak_idx]
            peaks.append(R)

        peaks = numpy.array(peaks)

        # Obtain reference lattice sites.
        R_shift = peaks[numpy.argmin(scipy.linalg.norm(peaks, axis=1))]
        return R_shift

    def get_shifted_lattice_sites_w_R_shift(self, grid, weight, bloch_coeff_grid):
        self.R_shift = self.get_R_shift(grid, weight, bloch_coeff_grid)
        self.sites = self.get_lattice_sites(R_shift=self.R_shift)
        return self.sites

    def get_shifted_lattice_sites(self, R_shift):
        self.R_shift = R_shift
        self.sites = self.get_lattice_sites(R_shift=self.R_shift)
        return self.sites

    def get_centered_grid(self, sites, grid):
        sites_center = numpy.mean(sites, axis=0)
        grid_center = numpy.mean(grid, axis=0)

        if self.verbose:
            print(f'\n# sites_center = {sites_center}')
            print(f'# grid_center = {grid_center}')
            print(f'# shift = {sites_center - grid_center}')

        grid += sites_center - grid_center
        return grid
    
    def fix_gauge(self, method='c0', R=None, C=0.01):
        if method == 'c0': 
            print('\n# Fixing gauge by requiring C_k[G = 0] \in R^+....') 
            self.mo_coeff = gf.fix_gauge_with_c0(self.mo_coeff)
            
        elif method == 'proj':
            print(f'\n# Fixing gauge by projecting Gaussian onto Bloch orbitals...')
            if R is None: 
                print(f'# Fixing Gaussian center to the default of (0, 0)...')
                R = numpy.zeros(2)
                
            Gvecs = self.basis @ self.recip_vecs_u.T
            self.mo_coeff = gf.fix_gauge_with_proj(self.mo_coeff, Gvecs, R, C, verbose=self.verbose)
            print(f'max|coeff.imag| = {numpy.amax(numpy.absolute(self.mo_coeff.imag))}')
    
    def get_transformation_matrix(self, sites):
        """
        Constructs the nkpt x nsite matrix U that defines the transformation
            C_wannier = C_bloch @ U,

        where C_* are coefficient matrices in the PW basis.
        """
        U = numpy.zeros((self.nsite, self.nsite), dtype=numpy.complex128)

        for ik, kpt in enumerate(self.kpts):
            for iR, site in enumerate(sites):
                U[ik, iR] = numpy.exp(-1.j * (kpt @ site))

        return U / numpy.sqrt(self.nsite)

    def get_wannier_coeff_grid(self, grid, weight, sites):
        """
        Constructs the Wannier functions in the grid basis.
        """
        X = _get_X(self.kpts, grid, self.area_u)
        Y = _get_Y(self.basis, self.recip_vecs_u, grid)
        U = self.get_transformation_matrix(sites)
        wannier_coeff_grid = numpy.einsum('rk,rG,Gk,kR->rR', X, Y, 
                                          self.mo_coeff, U, optimize=True)

        # Normalize.
        for iR in range(self.nsite):
            norm = numpy.sum(weight * numpy.absolute(wannier_coeff_grid[:, iR])**2)
            wannier_coeff_grid[:, iR] /= numpy.sqrt(norm)
        
        # Check.
        ovlp_grid = self.get_ovlp_grid(weight, wannier_coeff_grid)
        ovlp_diag = numpy.diag(ovlp_grid)
        ovlp_offdiag = ovlp_grid - numpy.diag(ovlp_diag)
        #numpy.testing.assert_allclose(ovlp_diag, 1., atol=1e-12)
        #numpy.testing.assert_allclose(ovlp_offdiag, 0., atol=1e-12)

        return wannier_coeff_grid

    def get_dm_grid(self, weight, coeff_grid):
        """
        Computes the density matrix in the grid basis.

        Returns
            dm : numpy.ndarray
                The density matrix of shape (ngrid, ngrid) in the grid basis.
        """
        return _get_dm_grid(weight, coeff_grid)

    def get_ovlp_grid(self, weight, coeff_grid):
        """
        Computes the overlap matrix for each `coeff`.

        Returns
            ovlp : numpy.ndarray
                The overlap matrix of shape (ncoeff, ncoeff).
        """
        return _get_ovlp_grid(weight, coeff_grid)
    
    def get_rho_grid(self, weight, coeff_grid):
        """
        Computes the charge density of each `coeff` over the grid.

        Returns
            rho : numpy.ndarray
                The matrix of shape (ngrid, ncoeff) storing the charge density 
                at each grid point, i.e. psi_i(r_g).
        """
        return _get_rho_grid(weight, coeff_grid) 


