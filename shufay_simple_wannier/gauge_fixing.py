import numpy
import scipy.special
import scipy.integrate

def get_gaussian_coeff(G, R, C):
    """
    Gaussian integral over an infinite area.
    """
    G2 = scipy.linalg.norm(G)**2
    exp = -(G2/(4.*C) + 1.j*G@R)
    return numpy.pi/C * numpy.exp(exp)

def get_gaussian_coeffs(Gvecs, R, C):
    """
    Construct Gaussian orbital in the unit cell. Uses Gaussian integrals over an infinite area.
    """
    nbsf = Gvecs.shape[0]
    gaussian_coeffs = numpy.zeros(nbsf, dtype=numpy.complex128)
    
    for iG, G in enumerate(Gvecs):
        gaussian_coeffs[iG] = get_gaussian_coeff(G, R, C)
    
    norm = scipy.linalg.norm(gaussian_coeffs)
    gaussian_coeffs /= norm
    return gaussian_coeffs

def fix_gauge_with_proj(bloch_coeffs, Gvecs, R, C, verbose=False):
    """
    Note that `bloch_coeffs` are Bloch orbitals in 1 band only. Uses Gaussian integrals over an infinite area.
    """
    gaussian_coeffs = get_gaussian_coeffs(Gvecs, R, C)
    nkpt = bloch_coeffs.shape[1]
    new_bloch_coeffs = numpy.zeros(bloch_coeffs.shape, dtype=bloch_coeffs.dtype)
    
    for ik in range(nkpt):
        bloch_coeffs_ik = bloch_coeffs[:, ik]
        
        # Projection.
        new_bloch_coeffs[:, ik] = numpy.dot(bloch_coeffs_ik.conj(), gaussian_coeffs) * bloch_coeffs_ik
        
        # Normalize.
        new_bloch_coeffs[:, ik] /= scipy.linalg.norm(new_bloch_coeffs[:, ik])
        
        if verbose:
            ovlp = numpy.dot(bloch_coeffs_ik.conj(), new_bloch_coeffs[:, ik])
            print(f'{ik}: {ovlp}')
            
    return new_bloch_coeffs

def fix_gauge_with_c0(bloch_coeffs):
    """
    Fix the gauge of Bloch wavefunctions by requiring C_k[G = 0] \in R^+.

    Follows the procedure in the Phoebe code:
    [https://phoebe.readthedocs.io/en/develop/theory/wannier.html#wannier-interpolation-of-band-structure]

        - For non-degenerate eigenvectors, the gauge is fixed by setting C_k[G = 0] \in R^+.

        - For degenerate eigenvectors, the gauge is fixed by setting C_k[G = 0] \in R^+ for 
          the first band only (since degeneracy also implies potential band crossings).
          Note: we don't have to do anything special for this since we already assume a 1-band model.
    """
    nkpt = bloch_coeffs.shape[1]
    phase_arr = []

    for ik in range(nkpt):
        c0 = bloch_coeffs[0, ik]
        phase_arr.append(numpy.angle(c0))

    phase_arr = numpy.array(phase_arr)
    new_bloch_coeffs = bloch_coeffs @ numpy.diag(numpy.exp(-1.j * phase_arr))

    # Check.
    for ik in range(nkpt):
        c0 = new_bloch_coeffs[0, ik]
        phase = numpy.angle(c0)
        assert numpy.isclose(c0.imag, 0)
        assert c0 > 0.
        
    return new_bloch_coeffs

def _get_gaussian_coeff(G, R, L, C):
    """
    Gaussian integral over a finite area.
    """
    diff = R - 1.j*G / (2.*C)
    diff2 = diff @ diff
    R2 = R @ R
    prefactor = numpy.exp(-C * (R2-diff2))
    
    def re_integrand(y):
        tan = numpy.tan(numpy.pi/3.)
        term1 = scipy.special.erf((diff[0] - y/tan) * numpy.sqrt(C))
        term2 = scipy.special.erf((diff[0] - L - y/tan) * numpy.sqrt(C))
        exp = -C * (y - diff[1])**2
        return numpy.real(exp * (term1 - term2))
    
    def im_integrand(y):
        tan = numpy.tan(numpy.pi/3.)
        term1 = scipy.special.erf((diff[0] - y/tan) * numpy.sqrt(C))
        term2 = scipy.special.erf((diff[0] - L - y/tan) * numpy.sqrt(C))
        exp = -C * (y - diff[1])**2
        return numpy.imag(exp * (term1 - term2))
    
    re_integral, re_err = scipy.integrate.quad(re_integrand, 0., L*numpy.sin(numpy.pi/3.))
    im_integral, im_err = scipy.integrate.quad(im_integrand, 0., L*numpy.sin(numpy.pi/3.))
    integral = re_integral + 1.j * im_integral
    err = numpy.absolute(re_err + 1.j * im_err)
    return  prefactor*integral, prefactor*err
    
def _get_gaussian_coeffs(Gvecs, R, L, C):
    """
    Construct Gaussian orbital in the unit cell. Uses Gaussian integrals over a finite area.
    """
    nbsf = Gvecs.shape[0]
    gaussian_coeffs = numpy.zeros(nbsf, dtype=numpy.complex128)
    errs = numpy.zeros(nbsf, dtype=numpy.complex128)
    
    for iG, G in enumerate(Gvecs):
        gaussian_coeffs[iG], errs[iG] = _get_gaussian_coeff(G, R, L, C)
    
    norm = scipy.linalg.norm(gaussian_coeffs)
    gaussian_coeffs /= norm
    errs /= norm
    print(f'max relative err = {numpy.amax(numpy.divide(errs, gaussian_coeffs))}')
    return gaussian_coeffs

def _fix_gauge_with_proj(bloch_coeffs, Gvecs, R, L, C, verbose=False):
    """
    Note that `bloch_coeffs` are Bloch orbitals in 1 band only. Uses Gaussian integrals over a finite area.
    """
    gaussian_coeffs = _get_gaussian_coeffs(Gvecs, R, L, C)
    nkpt = bloch_coeffs.shape[1]
    new_bloch_coeffs = numpy.zeros(bloch_coeffs.shape, dtype=bloch_coeffs.dtype)
    
    for ik in range(nkpt):
        bloch_coeffs_ik = bloch_coeffs[:, ik]
        
        # Projection.
        new_bloch_coeffs[:, ik] = numpy.dot(bloch_coeffs_ik.conj(), gaussian_coeffs) * bloch_coeffs_ik
        
        # Normalize.
        new_bloch_coeffs[:, ik] /= scipy.linalg.norm(new_bloch_coeffs[:, ik])
        
        if verbose:
            ovlp = numpy.dot(bloch_coeffs_ik.conj(), new_bloch_coeffs[:, ik])
            print(f'{ik}: {ovlp}')
            
    return new_bloch_coeffs
