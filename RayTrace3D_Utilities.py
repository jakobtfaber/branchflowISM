import numpy as np
from numpy.random import randn

from scipy.special import gamma
from scipy.special import airy
from scipy.special import ai_zeros
from scipy.interpolate import *
from scipy.fftpack import fft2, ifft2, fftshift, fft, ifft
from scipy.ndimage import map_coordinates

from matplotlib import cm
import matplotlib.pyplot as plt

# Define relevant constants
c = 2.998e10 #speed of light
pctocm = 3.0856e18 #1 pc in cm
GHz = 1e9 #central frequency
re = 2.8179e-13 #electron radius
kpc = 1e3 # in units of pc
autocm = 1.4960e13 #1 AU in cm
pi = np.pi
nscreen = 20

import sympy as sym

import multiprocessing
from numba import jit

# Generate 2D Gaussian Random Potential (in this case, a phase screen placed
# orthogonal to the line of sight).

def Gaussian(y_,r_, N=10, sigma=10):
    """
    Generate 2D Gaussian Random Potential as a superposition 
    of random plane waves
    """
    #Define random variables of plane waves
    theta = np.random.uniform(0, 2*np.pi, N+1)
    phi = np.random.uniform(0, 2*np.pi, N+1)

    return (sigma* np.sqrt(2/N)*(np.sum(np.cos(r_*np.sin(np.random.uniform(0, 2*np.pi)) \
                   + y_*np.cos(np.random.uniform(0, 2*np.pi)) + np.random.uniform(0, 2*np.pi)) for j in range(1, N+1))))

# First derivative w.r.t. y
def Gaussian_y(y_,r_, N=10, sigma=10):
    """
    First derivative of 2D Gaussian Random Potential
    """
    #Define random variables of plane waves
    theta = np.random.uniform(0, 2*np.pi, N+1)
    phi = np.random.uniform(0, 2*np.pi, N+1)

    return (sigma* np.sqrt(2/N)*(np.sum(-np.sin(r_*np.sin(np.random.uniform(0, 2*np.pi)) \
                    + y_*np.cos(np.random.uniform(0, 2*np.pi)) + \
                            np.random.uniform(0, 2*np.pi))*np.cos(np.random.uniform(0, 2*np.pi)) for j in range(1, N+1))))

# Generate 2D Kolmogorov turbulent phase screen

def Kolmogorov(nx, ny, dx=0.05, dy=0.05, rf = 1., psi=0, ar=1, alpha=5./3., mb2=2., inner=0.01):
    
    """
    Get 2D Kolmogorov turbulent phase screen
        
    mb2: Max Born parameter for strength of scattering
    rf: Fresnel scale
    ds (or dx,dy): Spatial step sizes with respect to rf
    alpha: Structure function exponent (Kolmogorov = 5/3)
    ar: Anisotropy axial ratio
    psi: Anisotropy orientation
    inner: Inner scale w.r.t rf - should generally be smaller than ds
    ns (or nx,ny): Number of spatial steps
    nf: Number of frequency steps.
    dlam: Fractional bandwidth relative to centre frequency
    lamsteps: Boolean to choose whether steps in lambda or freq
    seed: Seed number, or use "-1" to shuffle
        
    """

    randseed = np.random.randint(1235, 135567)
    #print('Kolmogorov Phase Screen Generated')
    #print('Random Seed: ', randseed)
    np.random.seed(randseed)

    nx2 = int(nx/2 + 1)
    ny2 = int(ny/2 + 1)

    #Initialize array
    w = np.zeros([nx, ny])  
    dqx = 2*np.pi/(dx*nx)
    dqy = 2*np.pi/(dy*ny)
    
    ns = 1
    a2 = alpha*0.5
    aa = 1.0+a2
    ab = 1.0-a2
    cmb2 = alpha*mb2 / (4*np.pi * gamma(ab)*np.cos(alpha * np.pi*0.25)*ns)
    consp = cmb2*dqx*dqy/(rf**alpha)
    
    def discrete_sample_spectrum(kx=0, ky=0):
        
        cs = np.cos(psi*np.pi/180)
        sn = np.sin(psi*np.pi/180)
        r = ar
        con = np.sqrt(consp)
        alf = -(alpha+2)/4
        # anisotropy parameters
        a = (cs**2)/r + r*sn**2
        b = r*cs**2 + sn**2/r
        c = 2*cs*sn*(1/r-r)
        q2 = a * np.power(kx, 2) + b * np.power(ky, 2) + c*np.multiply(kx, ky)
        # isotropic inner scale
        out = con*np.multiply(np.power(q2, alf),
                              np.exp(-(np.add(np.power(kx, 2),
                                              np.power(ky, 2))) *
                                                     inner**2/2))
        return out

    #Screen Weights
    # first do ky=0 line
    k = np.arange(2, nx2+1)
    w[k-1, 0] = discrete_sample_spectrum(kx=(k-1)*dqx, ky=0)
    w[nx+1-k, 0] = w[k, 0]
    # then do kx=0 line
    ll = np.arange(2, ny2+1)
    w[0, ll-1] = discrete_sample_spectrum(kx=0, ky=(ll-1)*dqy)
    w[0, ny+1-ll] = w[0, ll-1]
    # now do the rest of the field
    kp = np.arange(2, nx2+1)
    k = np.arange((nx2+1), nx+1)
    km = -(nx-k+1)
    for il in range(2, ny2+1):
        w[kp-1, il-1] = discrete_sample_spectrum(kx=(kp-1)*dqx, ky=(il-1)*dqy)
        w[k-1, il-1] = discrete_sample_spectrum(kx=km*dqx, ky=(il-1)*dqy)
        w[nx+1-kp, ny+1-il] = w[kp-1, il-1]
        w[nx+1-k, ny+1-il] = w[k-1, il-1]

    #Generate Complex Gaussian Array
    gaussarr = np.multiply(w, np.add(randn(nx, ny),
                                1j*randn(nx, ny)))

    scr = np.real(fft2(gaussarr))
    
    return scr

def Kolmogorov_cordes(ux, uy, inner=0.01, outer=1000, phiF_r=20., dx=0.05, dy=0.05, rF=1., alpha=5./3., normfres=False):

    
    """
    Generates npoints of a realization of power-law noise with unit 
    variance with spectral index si and inner and outer scales as 
    specified for a sample interval dx. 
    
    OR
    
    Generates npoints of a realization of gaussian noise.

    input:
    si = spectral index of power-law wavenumber spectrum
    phiF = rms phase at Fresnel scale (rad)
        length scales: all dimensionless:
        rF      = Fresnel scale
        inner, outer    = inner and outer scales
        dx, dy      = sample intervals
        xwidth, ywidth  = screen extent
    logical:
            normfres    = True implies normalization to phiF

        Definition of Fresnel scale: r_F^2 = \lambda D / 2\pi

    returns:
       xvec, yvec, xseries, xseries_norm, qxvec, qyvec, qshape
       (xvec, yvec) coordinates perpendicular to line of sight 
       xseries = screen phase 
       xseries_norm = screen phase scaled to input rms phase on Fresnel scale
       qxvec, qyvec  = wavenumber axes
       qshape = sqrt(shape of wavenumber spectrum)

    """
    
    #specified diffraction and refraction scales
    lr = rF * phiF_r 

    nx = int(abs(ux/dx))
    ny = int(abs(uy/dy))
    xvec = (np.arange(0.,nx)-nx/2+1)*dx
    yvec = (np.arange(0.,ny)-ny/2+1)*dy
    
    dqx = 2.*np.pi / ux
    dqy = 2.*np.pi / uy
    qmaxx = (2.*np.pi) / (2.*dx)
    qmaxy = (2.*np.pi) / (2.*dy)
    
    nqx = 2*int(qmaxx/dqx)
    nqy = 2*int(qmaxy/dqy)
    nqx = nx
    nqy = ny
    qxvec = (np.arange(0.,nqx)-nqx/2+1)*dqx
    qxvec = np.roll(qxvec,nqx//2+1)
    qyvec = (np.arange(0.,nqy)-nqy/2+1)*dqy
    qyvec = np.roll(qyvec,nqy//2+1)
    
    #lower wavernumber cutoff for phase power spectrum
    qin = 2.*np.pi / inner
    
    #upper wavernumber cutoff for phase power spectrum
    qout = 2.*np.pi / outer
    
    qshape = np.zeros((nqx, nqy))
    qshape_rolloff = np.zeros((nqx, nqy))

    qmax = qxvec.max()/2.
    for i, qxi in np.enumerate(qxvec):
        for j, qyj in np.enumerate(qyvec):
            qsq = qxi**2 + qyj**2
            qshape[i,j] = (qout**2 + qsq)**(-alpha/4.) * np.exp(-qsq/(2.*qmax**2))
                
    npoints = np.size(qshape)
    #print('2D Npoints: ', npoints)
    xformr=np.randn(nqx, nqy)*qshape
    xformi=np.randn(nqx, nqy)*qshape
    xform = xformr + 1j*xformi
    #print('2D Gridshape: ', np.shape(xform))
    spectrum = np.abs(xform)**2
    xseries = np.real(ifft2(xform))

    return xseries


# Define functions for the lens equation x' = x + alpha * nabla(psi(x,y))
# where alpha = (rF**2 * phi_0), phi_0 being the strength of the lens,
# and psi is an arbitrary 2D function with unit maximum that describes 
# the shape of the electron density variation in the lens plane

def alpha(dso, dsl, f, dm):
    """ 
    Parameters: 
        dso : distance from source to observer
        dsl : distance from source to lens
        f   : frequency
        dm  : dispersion measure (max electron 
                density perturation in pc cm**-3)
    Returns:
        alpha coefficient : alpha : rF**2 * phi0,
                                    phi0 : lens strength
    """
    dlo = dso - dsl #distance from lens to observer
    return -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso)

def rFsqr(dso, dsl, f):
    """ 
    Parameters: 
        dso : distance from source to observer
        dsl : distance from source to lens
        f   : frequency
        dm  : dispersion measure (max electron 
                density perturation in pc cm**-3)
    Returns:
        rF**2 : Square of the Fresnel scale rF
    """
    dlo = dso - dsl #distance from lens to observer
    rF = np.sqrt(c*dsl*dlo/(2*pi*f*dso))
    return rF**2

def lensc(dm, f):
    """ 
    Paramters:
        f   : frequency
        dm  : dispersion measure (max electron
    Returns:
        phi0 : Coefficient that determines the phase perturbation due to the lens 
    """
    phi0 = -c*re*dm/f
    return phi0

def tg0coeff(dso, dsl):
    """ 
    Parameters: 
        dso : distance from source to observer
        dsl : distance from source to lens
    Returns:
        Normalization coefficient of some kind???
    """
    dlo = dso - dsl
    return dso/(2*c*dsl*dlo)

def tdm0coeff(dm, f):
    '''
    Paramters:
        f   : frequency
        dm  : dispersion measure (max electron
    Returns:
        Normalization coefficient of some kind???
    '''
    return c*re*dm/(2*pi*f**2)

# Phase
def phi(uvec, rF2, lc, ax, ay, V=None):
    """ Returns the phase at a stationary point. """
    ux, uy = uvec
    grad = np.asarray(np.gradient(V))
    #grad = np.asarray(np.gradient(V))
    return 0.5*rF2*lc**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + lc*V - 0.5*pi

# Field
def field(uvec, rF2, lc, ax, ay, V=None):
    """ Returns the elements of the geometrical optics field: the amplitude and the phase, including the phase shift as determined by the sign of the derivatives. """
    ux, uy = uvec
    alp = rF2*lc
    #Vx = np.gradient(V, axis=1)
    #Vxx = np.gradient(Vx, axis=1)
    #Vxy = np.gradient(Vx, axis=0)
    #Vy = np.gradient(V, axis=0)
    #Vyy = np.gradient(Vy, axis=0)
    #psi20, psi02, psi11 = np.array([Vxx, Vyy, Vxy])
    psi20, psi02, psi11 = lensh(ux, uy)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    sigma = np.sign(phi02)
    H = phi20*phi02 - phi11**2
    delta = np.sign(H)
    amp = (ax*ay/rF2)*np.abs(H)**-0.5
    phase = phi(uvec, rF2, lc, ax, ay, V)
    pshift = pi*(delta + 1)*sigma*0.25
    # return amp*np.exp(1j*(phase + pshift))
    #field_  = amp*np.exp(1j*(phase + pshift))

    return amp, phase, pshift

def mapToUprime(uvec, alp, ax, ay, rF2, lc = 1, V=None):
    """ 
    Parameters:
        uvec : vector containing u-plane coordinates
        alp : alpha coefficient
        ax : characteristic length scale in x
        ay : characteristic length scale in y
    Returns:   
        [upx, upy] : array of coordinates in the u-plane that have been
                    mapped to coordiantes in the u'-plane
    """
    ux, uy = uvec
    #grad = np.asarray(np.gradient(V))
    grad = lensg(ux, uy)
    upx = ux + alp*grad[0]/ax**2
    upy = uy + alp*grad[1]/ay**2
    rays = np.array([upx, upy])

    # Calculate field (amp, phase, phaseshift)
    amp, phase, pshift = field(uvec, rF2, lc, ax, ay, V)

    return rays, phase, amp, pshift

def compLensEq(uvec, upvec, coeff):
    """ 
    Evaluates the 2D lens equation with u a complex vector. 
    """
    uxr, uxi, uyr, uyi = uvec
    upx, upy = upvec
    grad = lensg(*[uxr + 1j*uxi, uyr + 1j*uyi])
    eq = np.array([uxr + 1j*uxi + coeff[0]*grad[0] - upx, uyr + 1j*uyi + coeff[1]*grad[1] - upy])
    return [eq[0].real, eq[0].imag, eq[1].real, eq[1].imag]

def lensEq(uvec, upvec, coeff, V):
    """ 
    Evaluates the 2D lens equation. coeff = alp*[1/ax**2, 1/ay**2]. 
    """
    ux, uy = uvec
    upx, upy = upvec
    grad = np.asarray(np.gradient(V))
    return np.array([ux + coeff[0]*grad[0] - upx, uy + coeff[1]*grad[1] - upy])
    
def lensEqHelp(uvec, coeff, V):
    """ 
    Returns invariant of the lens equation. Coeff = alpp*[1./ax**2, 1./ay**2]. 
    """
    ux, uy = uvec
    #grad = lensg(ux, uy)
    grad = np.asarray(np.gradient(V))
    return np.array([coeff[0]*grad[0], coeff[1]*grad[1]])