import scipy
import numpy
from numpy.typing import NDArray
from numba import jit, njit

# Custom modules.
import lib
import utils
from planewaves import Basis
from kpoints import KPoints
from constants import *

class LocalizedOrbitals():
    def __init__(
            self, 
            mo_coeff: NDArray, 
            basis_obj: Basis, 
            nsite : int,
            verbose: bool = False):
        """
        Args
            mo_coeff : numpy.ndarray
                Coefficient matrix for wavefunctions in the plane wave basis.

            basis_obj : Basis
                `Basis` object storing quantities pertaining to the plane wave basis.

            R_shift : numpy.ndarray
                Position of a lattice site within the unit cell.
        """
        self.mo_coeff = mo_coeff # PW basis.
        
        # Check overlaps.
        ovlp = self.mo_coeff.T.conj() @ self.mo_coeff
        ovlp_diag = numpy.diag(ovlp)
        ovlp_offdiag = ovlp - numpy.diag(ovlp_diag)

        self.verbose = verbose
        self.basis = basis_obj.basis
        self.nbsf = self.basis.shape[0]
        self.gmax = numpy.amax(self.basis)
        
        # Supercell.
        self.nsite = nsite
        self.recip_vecs_s = basis_obj.recip_vecs # Recip. vecs; units: 1/Bohr
        self.direct_vecs_s = 2 * numpy.pi * scipy.linalg.inv(self.recip_vecs_s.T)
        self.L_s = scipy.linalg.norm(self.direct_vecs_s[:, 0]) # Period; units: Bohr
        self.area_s = abs(scipy.linalg.det(self.direct_vecs_s)) # Area; units: Bohr^2
        
        # Unit cell.
        self.nsitex = numpy.rint(numpy.sqrt(nsite)).astype(int)
        self.recip_vecs_u = self.recip_vecs_s * self.nsitex
        self.direct_vecs_u = self.direct_vecs_s / self.nsitex
        self.L_u = scipy.linalg.norm(self.direct_vecs_u[:, 0]) # Period; units: Bohr
        self.area_u = abs(scipy.linalg.det(self.direct_vecs_u)) # Area; units: Bohr^2

    def get_lattice_sites(self, R_shift=numpy.zeros(2)):
        """
        Construct array of lattice sites over the supercell.
        """
        if self.verbose:
            print(f'\n# Obtaining lattice sites with R_shift: \n{R_shift}\n')

        self.sites, _ = utils.get_lattice_sites(
                            self.nsite, self.direct_vecs_u, R_shift=R_shift)
        assert self.sites.shape[0] == self.nsite
        return self.sites

    def get_grid(self, ngridx=None, ncellx=1):
        """
        Construct array of real space grid points over the supercell.
        """
        if ngridx is None: ngridx = 2*self.gmax + 1
        grid, weight = utils.get_grid(self.direct_vecs_s, ngridx, ncellx=ncellx)
        return grid, weight
    

class KLocalizedOrbitals(LocalizedOrbitals):
    def __init__(
            self, 
            mo_coeff: NDArray, 
            basis_obj: Basis, 
            kpt_obj: KPoints,
            verbose: bool = False):
        """
        Args
            mo_coeff : numpy.ndarray
                Coefficient matrix for wavefunctions in the plane wave basis.

            basis_obj : Basis
                `Basis` object storing quantities pertaining to the plane wave 
                basis and sampled k-points.

            kpt_obj : KPoints
                `KPoints` object storing quantities pertaining to the sampled k-points.
        """
        self.mo_coeff = mo_coeff # PW basis.
        self.kpts = kpt_obj.kpts
        
        self.verbose = verbose
        self.basis = basis_obj.basis
        self.nbsf = self.basis.shape[0]
        self.gmax = numpy.amax(self.basis)
        
        # Unit cell.
        self.recip_vecs_u = basis_obj.recip_vecs # Recip. vecs; units: 1/Bohr
        self.direct_vecs_u = 2 * numpy.pi * scipy.linalg.inv(self.recip_vecs_u.T)
        self.L_u = scipy.linalg.norm(self.direct_vecs_u[:, 0]) # Period; units: Bohr
        self.area_u = abs(scipy.linalg.det(self.direct_vecs_u)) # Area; units: Bohr^2
        
        # Supercell.
        self.nsite = self.kpts.shape[0]
        self.nsitex = numpy.rint(numpy.sqrt(self.nsite)).astype(int)
        self.recip_vecs_s = self.recip_vecs_u / self.nsitex
        self.direct_vecs_s = self.direct_vecs_u * self.nsitex
        self.L_s = scipy.linalg.norm(self.direct_vecs_s[:, 0]) # Period; units: Bohr
        self.area_s = abs(scipy.linalg.det(self.direct_vecs_s)) # Area; units: Bohr^2
        
