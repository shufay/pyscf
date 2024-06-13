import sys
import scipy
import numpy
from numba import njit

# Custom modules.
import lib

def _dot_ao_dm(ao, dm):
    """
    Return numpy.dot(ao, dm).
    """
    ngrid, nao = ao.shape
    return numpy.dot(ao, numpy.asarray(dm, order='C'))

def _dot_ao_ao(ao1, ao2):
    """
    Return numpy.dot(ao1.T.conj(), ao2).
    """
    ngrid, nao = ao1.shape
    return numpy.dot(ao1.T.conj(), ao2)

def _contract_rho(bra, ket):
    bra = bra.T
    ket = ket.T
    rho = numpy.einsum('ip,ip->p', bra.real, ket.real)
    rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    return rho

@njit(fastmath=True)
def eval_ao(kvecs, grid, area):
    """ 
    Evaluate AO function value on the given grid. We only consider plane waves given by

        Phi_G(r) = exp[IGr] / sqrt[Omega].

    Args:
        kvecs : (nao, 2) ndarray
            Array of reciprocal vectors in the plane wave basis.

        grid : (ngrid, 2) ndarray
            Array of grid point coordinates.

        area : float
            Area of unit cell (Bohr^2).

    Returns:
        Array of shape (ngrid, nao) of AO function values on the grid.

    Example:
    >>> kvecs = basis @ recip_vecs.T
    >>> grid = numpy.random.rand((100, 2))
    >>> ao = eval_ao(kvecs, grid, ham.area)
    """
    exponents = 1j * (grid @ kvecs.T)
    return numpy.exp(exponents) / numpy.sqrt(area)

def eval_rho(ao, dm):
    """
    Calculate the electron density given a density matrix.
    """
    ngrid, nao = ao.shape
    c0 = _dot_ao_dm(ao, dm) # (ngrid, nmo) array
    rho = _contract_rho(ao, c0)
    return numpy.real(rho)
    
def get_grid(direct_vecs, ngridx, ncellx=1, center=False, index=False):
    area = abs(scipy.linalg.det(direct_vecs)) # Area; units: Bohr^2
    idx = lib.cartesian_prod(2 * [numpy.arange(ngridx)])
    division = ncellx * direct_vecs / ngridx
    grid = idx @ division.T
    
    if center: grid -= numpy.mean(grid, axis=0)

    if index: # Make skewed grid regular.
        L = scipy.linalg.norm(direct_vecs[:,0]) # Period; units: Bohr
        division = ncellx * L / ngridx
        grid = idx * division

    weight = ncellx**2 * area / grid.shape[0]
    return numpy.array(grid, order='C'), weight

def get_lattice_sites(nsite, direct_vecs, R_shift=numpy.zeros(2)):
    nsitex = numpy.sqrt(nsite) # Number of divisions along each axis.
    idx = lib.cartesian_prod([numpy.arange(nsitex), numpy.arange(nsitex)]) 
    sites = idx @ direct_vecs.T + R_shift
    return sites, idx
