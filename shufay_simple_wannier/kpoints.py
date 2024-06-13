import scipy
import numpy
from numpy.typing import NDArray

# Custom modules.
import lib 

def get_perpendicular_bisector(v):
    midpoint = v/2.
    normal = numpy.array([-v[1], v[0]])
    return midpoint, normal

class KPoints():
    """
    Class to hold properties pertaining to sampled k-points. 
    """
    def __init__(
            self, 
            recip_vecs : NDArray, 
            ncellx : int = None, 
            kpts : NDArray = None, 
            opt : str ='mdpm', 
            wrap_around : bool = False,
            with_gamma_point : bool = True,
            verbose : bool = False):
        # Constants.
        self.CORNER_MULTIPLICITY = 3
        self.EDGE_MULTIPLICITY = 2
        self.verbose = verbose

        self.recip_vecs = recip_vecs
        self.first_shell = self.get_first_shell()

        # Get BZ edges and corners.
        self.bz_edges, self.midpoints, self.normals = self.get_bz_edges()
        self.bz_corners = self.get_bz_corners()

        # K-point sampling.
        self.kpts = kpts

        if self.kpts is None:
            if opt == 'mdpm':
                self.kpts, _, _, _ = self.get_kpts_mdpm(ncellx)

            elif opt == 'mp':
                self.kpts, _, _ = self.get_kpts_mp(
                        ncellx, wrap_around=wrap_around, with_gamma_point=with_gamma_point)

        self.nkpt = self.kpts.shape[0]

    def get_first_shell(self):
        """
        Get the first shell of reciprocal vectors stored as rows of a matrix.
        """
        recip_vecs = self.recip_vecs
        theta = numpy.pi / 3.
        nrot = 5
        rotmat = numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                              [numpy.sin(theta),  numpy.cos(theta)]])
        G = recip_vecs[:, 0]
        Gs = [G]
        
        for i in range(nrot):
            G_last = G
            G = rotmat @ G_last
            Gs.append(G)
        
        Gs = numpy.array(Gs)
    
        # Arrange in order of increasing angle in the range [-pi, pi].
        angles = numpy.arctan2(Gs[:, 1], Gs[:, 0])
        sort_idx = numpy.argsort(angles)
        return Gs[sort_idx]

    def get_bz_edges(self, tmax=1):
        t = numpy.linspace(-tmax, tmax, 100)
        ms = []
        ns = []
        bz_edges = []
        
        for G in self.first_shell:
            m, n = get_perpendicular_bisector(G)
            bz_edge = numpy.outer(t, n) + m
            ms.append(m)
            ns.append(n)
            bz_edges.append(bz_edge)

        ms = numpy.array(ms)
        ns = numpy.array(ns)
        bz_edges = numpy.array(bz_edges)
        return bz_edges, ms, ns

    def get_bz_corners(self):
        first_shell = self.first_shell
        corners = []
        
        for i in range(6):
            g = first_shell[i] + first_shell[(i+1)%6]
            g /= scipy.linalg.norm(g)
            corners.append(g)
        
        corners = numpy.array(corners)
        gnorm = scipy.linalg.norm(first_shell[0])
        edge_norm = gnorm / 2.
        corner_norm = edge_norm / numpy.cos(numpy.pi / 6.)
        return corners * corner_norm
    
    def is_in_bz(self, kpt):
        midpoints = self.midpoints
        midpoint_norm = scipy.linalg.norm(midpoints[0])
        
        for i, midpoint in enumerate(midpoints):
            proj = kpt @ midpoint / midpoint_norm
            
            if (proj > midpoint_norm):
                if numpy.isclose(proj, midpoint_norm):
                    continue
                    
                return False
            
        return True

    def is_on_edge(self, kpt):
        dot_arr = []
        
        for i, midpoint in enumerate(self.midpoints):
            normal = self.normals[i]
            kmm = kpt - midpoint
            kmm_norm = scipy.linalg.norm(kmm)
            
            if numpy.isclose(kmm_norm, 0.): 
                return True
                
            kmm /= kmm_norm
            n = normal / scipy.linalg.norm(normal)
            dot_arr.append(kmm @ n)
            
        _is_on_edge = numpy.isclose(numpy.absolute(dot_arr), 1)
        return numpy.any(_is_on_edge)

    def is_corner(self, kpt):
        _is_corner = numpy.isclose(scipy.linalg.norm(self.bz_corners - kpt, axis=1), 0.)
        return numpy.any(_is_corner)

    def is_duplicate(self, kpt, kpts=None):
        if kpts is None: kpts = self.kpts

        if len(kpts) > 0:
            shifts = kpt + self.first_shell

            for shift in shifts:
                cond = numpy.isclose(scipy.linalg.norm(kpts - shift, axis=1), 0.)

                if numpy.any(cond): return True
            
        return False

    def get_outer_kpts(self, nshell, gs):
        # WHAT IS GS?!
        outer = []
        gnorm = scipy.linalg.norm(gs[:, 0])
        
        for kpt in self.kpts:
            knorm = scipy.linalg.norm(kpt)

            if knorm > (nshell-0.5) * gnorm:
                outer.append(kpt)
                    
        return numpy.array(outer)
    
    def get_kpts_mdpm(self, ncellx):
        recip_vecs = self.recip_vecs
        Gs = self.first_shell
        midpoints = self.midpoints
        nkpt = ncellx**2

        Gnorm = scipy.linalg.norm(recip_vecs[:, 0])
        bz_corners_norm = scipy.linalg.norm(self.bz_corners[0])
        mnorm = scipy.linalg.norm(midpoints[0])
        
        gs = recip_vecs / ncellx
        g1 = gs[:, 0]
        g2 = gs[:, 1]
        
        nmax = numpy.rint(mnorm / scipy.linalg.norm(g1)).astype(int) + 2
        ns = numpy.arange(-nmax, nmax+1)
        N = len(ns)
        kpts = []
        kpts_idx = []
        knorms = []
        nedge = 0
        ncorner = 0
        
        for i in range(N):
            for j in range(N):
                i1 = ns[i] 
                i2 = ns[j]
                kpt = i1*g1 + i2*g2
                knorm = scipy.linalg.norm(kpt)
                
                # In the mBZ.
                if self.is_in_bz(kpt):
                    # On the edge.
                    if self.is_on_edge(kpt):
                        if self.is_duplicate(kpt, kpts=kpts):
                            continue
                            
                        nedge += 1
                        
                    # On the corner.
                    if self.is_corner(kpt):
                        ncorner += 1
                        
                    kpts.append(kpt)
                    kpts_idx.append([i1, i2])
                    knorms.append(knorm)
            
        assert ncorner <= 2
        kpts = numpy.array(kpts)
        kpts_idx = numpy.array(kpts_idx)
        knorms = numpy.array(knorms)
        nkpt = len(kpts)

        # Sort kpts in the order of increasing energy.
        sort_idx = numpy.argsort(knorms)
        kpts = kpts[sort_idx]
        kpts_idx = kpts_idx[sort_idx]
        nshell = numpy.amax(kpts_idx[numpy.isclose(kpts_idx[:, 0], 0.)])
       
        if self.verbose:
            print(f'\n# nmax = {nmax}')
            print(f'# nkpt = {nkpt}')
            print(f'# nedge = {nedge}')
            print(f'# ncorner = {ncorner}')
            print(f'# nshell = {nshell}')

        return kpts, kpts_idx, gs, nshell

    def get_kpts_mp(self, ncellx, wrap_around=False, with_gamma_point=True):
        recip_vecs = self.recip_vecs
        ks = []
        
        if with_gamma_point:
            ks = numpy.arange(ncellx, dtype=float) / ncellx

        else:
            ks = (numpy.arange(ncellx)+0.5) / ncellx - 0.5

        if wrap_around:
            ks[ks >= 0.5] -= 1
            
        kpts_idx = lib.cartesian_prod(2*[ks])
        kpts = kpts_idx @ recip_vecs.T
        knorms = scipy.linalg.norm(kpts, axis=1)
        gs = recip_vecs / ncellx
        
        # Sort kpts in the order of increasing energy.
        sort_idx = numpy.argsort(knorms)
        kpts = kpts[sort_idx]
        kpts_idx = kpts_idx[sort_idx] * ncellx
        return kpts, kpts_idx, gs
    
    def process_kpts(self):
        self.bz_edge_idx = [] # Index of kpts on the BZ edge.
        self.bz_corner_idx = [] # Index of kpts on the BZ corners.
        
        for ik, k in enumerate(self.kpts):
            if self.is_corner(k):
                self.bz_corner_idx.append(ik)
            
            # Is not a corner if arrives here.
            elif self.is_on_edge(k):
                self.bz_edge_idx.append(ik)                
                
        self.ncorner = len(self.bz_corner_idx)
        self.nedge = len(self.bz_edge_idx)
        self.nouter = self.nedge + self.ncorner
        
        self.ncorner_unique = self.ncorner
        self.nedge_unique = self.nedge
    
        if self.ncorner_unique > 2: # `ncorner` is non-unique.
            self.ncorner_unique = numpy.rint(self.ncorner / self.CORNER_MULTIPLICITY).astype(int)
            self.nedge_unique = numpy.rint(self.nedge / self.EDGE_MULTIPLICITY).astype(int)
        
        self.nouter_unique = self.nedge_unique + self.ncorner_unique
        ninner = self.nkpt - self.nouter
        self.nkpt_unique = ninner + self.nouter_unique
        
        if self.verbose:
            print(f'\n# ncorner = {self.ncorner}')
            print(f'# nedge = {self.nedge}')
            print(f'# nouter = {self.nouter}')

            print(f'\n# ncorner_unique = {self.ncorner_unique}')
            print(f'# nedge_unique = {self.nedge_unique}')
            print(f'# nouter_unique = {self.nouter_unique}')

            print(f'\n# nkpt_unique = {self.nkpt_unique}')
            print(f'# ninner = {ninner}')
                
