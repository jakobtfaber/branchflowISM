
import numpy as np
from numpy.random import randn

from scipy.special import gamma as gfunc
from scipy.special import airy
from scipy.special import ai_zeros
from scipy.interpolate import *
from scipy.fftpack import fft2, ifft2, fftshift, fft, ifft
from scipy.ndimage import map_coordinates, filters
from scipy.signal import convolve2d, correlate2d, correlate, find_peaks, peak_widths
from scipy.stats import gaussian_kde
from scipy.spatial.distance import *
from scipy.interpolate import *

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import colors 
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
from IPython.display import Video

from RayTrace3D_Utilities import *

import sympy as sym
from sympy.printing.pycode import NumPyPrinter
from sympy.printing import ccode
from sympy import lambdify, Sum

import multiprocessing
from numba import jit
import builtins as bt
import math
from shapely import geometry

# Define relevant constants
c = 2.998e10 #speed of light
pctocm = 3.0856e18 #1 pc in cm
GHz = 1e9 #central frequency
re = 2.8179e-13 #electron radius
kpc = 1e3 # in units of pc
autocm = 1.4960e13 #1 AU in cm
pi = np.pi

# Define starting coordinates on source plane (x,y) in dimensionless coordinates (u_x, u_y)
# where u_x = (x / a_x) and a_x is the characteristic length scale (same goes for u_y).
# This is done using Sympy
u_x, u_y = sym.symbols('u_x u_y')
A, B = 1.5e-2, 5

#Use Sympy to find derivatives of the potential
N, j, theta, phi, sigma = sym.symbols('N j theta phi sigma') #name variables

# Define various lens geometries

gaussrand = sigma * sym.sqrt(2/N) * sym.Sum(sym.cos(u_y*sym.cos(sym.Indexed(theta, j)) + \
                            u_x*sym.sin(sym.Indexed(theta, j)) + sym.Indexed(phi, j)), (j, 1, N))
#gaussrand = 2 * sym.sqrt(2) * sym.cos(u_y*sym.cos(0.25*np.pi) + u_x*sym.sin(0.25*np.pi) + 2*np.pi)
sin = sym.sin(u_x+u_y)
gauss = sym.exp(-u_x**2-u_y**2) #gaussian
ring = 2.7182*(u_x**2 + u_y**2)*gauss #ring
rectgauss = sym.exp(-u_x**4-u_y**4)
stgauss = gauss*(1. - A*(sym.sin(B*(u_x))+sym.sin(B*(u_y - 2*pi*0.3)))) #rectangular gaussian
asymgauss = sym.exp(-u_x**2-u_y**4) #asymmetrical gaussian
supergauss2 = sym.exp(-(u_x**2+u_y**2)**2) #gaussian squared
supergauss3 = sym.exp(-(u_x**2+u_y**2)**3) #gaussian cubed
superlorentz = 1./((u_x**2 + u_y**2)**2+1.) #lorentzian with width (gamma) of 2

# Define preferred lens geometry (use gauss as test case)
lensfunc = sin

# Differentiate the lens equation to 1st, 2nd, and 3rd order using Sympy
#lensg = np.array([sym.diff(lensf, u_x), sym.diff(lensf, u_y)])
lensg = np.array([sym.diff(lensfunc, u_x), sym.diff(lensfunc, u_y)])
lensh = np.array([sym.diff(lensfunc, u_x, u_x), sym.diff(lensfunc, u_y, u_y), sym.diff(lensfunc, u_x, u_y)])
lensgh = np.array([sym.diff(lensfunc, u_x, u_x, u_x), \
                sym.diff(lensfunc, u_x, u_x, u_y), \
                   sym.diff(lensfunc, u_x, u_y, u_y), \
                       sym.diff(lensfunc, u_y, u_y, u_y)])

# Use Sympy to turn the lens equations into Numpy functions using Sympy
lensfun = sym.lambdify([u_x, u_y], lensfunc, 'numpy')
lensg = sym.lambdify([u_x, u_y], lensg, 'numpy')
lensh = sym.lambdify([u_x, u_y], lensh, 'numpy')
lensgh = sym.lambdify([u_x, u_y], lensgh, 'numpy')

#Gaussian screen functions & derivatives

scrfun = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.cos(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j]) for j in range(1, N-1))
scrgx = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(-np.sin(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j])*np.sin(theta[j]) for j in range(1, N-1))
scrgy = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(-np.sin(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j])*np.cos(theta[j]) for j in range(1, N-1))
scrgxx = lambda u_x, u_y, theta, phi, N, sigma : \
        -np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(theta[j])**2*np.cos(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j]) for j in range(1, N-1))
scrgyy = lambda u_x, u_y, theta, phi, N, sigma : \
        -np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.cos(u_x*np.sin(theta[j]) + u_y*np.cos(theta[j]) + \
                phi[j])*np.cos(theta[j])**2  for j in range(1, N-1))
scrgxy = lambda u_x, u_y, theta, phi, N, sigma : \
        -np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(theta[j])*np.cos(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j])*np.cos(theta[j]) for j in range(1, N-1))
scrgxxx = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(u_x*np.sin(theta[j]) + u_y*np.cos(theta[j]) + \
            phi[j])*np.sin(theta[j])**3 for j in range(1, N-1))
scrgxxy = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(u_x*np.sin(theta[j]) + u_y*np.cos(theta[j]) + \
                    phi[j])*np.sin(theta[j])**2*np.cos(theta[j]) for j in range(1, N-1))
scrgxyy = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(u_x*np.sin(theta[j]) + u_y*np.cos(theta[j]) + \
                        phi[j])*np.sin(theta[j])*np.cos(theta[j])**2 for j in range(1, N-1))
scrgyyy = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(u_x*np.sin(theta[j]) + u_y*np.cos(theta[j]) + \
                        phi[j])*np.cos(theta[j])**3 for j in range(1, N-1))
                                             

def mapToUprime(uvec, alp, ax, ay, rF2, lc, sigma, theta, phi, N, V=None):
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
    
    #scr = V
    #V10 = np.gradient(V, axis = 0)
    #V01 = np.gradient(V, axis = 1)
    #V20 = np.gradient(V10, axis = 0)
    #V02 = np.gradient(V01, axis = 1)
    #V11 = np.gradient(V10, axis = 1)
    #V30 = np.gradient(V20, axis = 0)
    #V03 = np.gradient(V02, axis = 1)
    #V21 = np.gradient(V20, axis = 1)
    #V12 = np.gradient(V02, axis = 0)
 
    #upx = ux + alp*V10/ax**2
    #upy = uy + alp*V01/ay**2
    #rays = np.array([upx, upy])
 
    ## Calculate Amplitude, Field, Phase and Phase Shift
    #alp = rF2*lc
    #psi20 = V20
    #psi02 = V02
    #psi11 = V11
    #phi20 = ax**2/rF2 + lc*psi20
    #phi02 = ay**2/rF2 + lc*psi02
    #phi11 = lc*psi11
    #H = phi20*phi02 - phi11**2
    #sigma = np.sign(phi02)
    #delta = np.sign(H)
    #amp = (ax*ay/rF2)*np.abs(H)**-0.5
 
    #phase = 0.5*rF2*lc**2*((V10/ax)**2 + \
    #                    V01**2) + \
    #                    lc*V - 0.5*pi
 
    #pshift = pi*(delta + 1)*sigma*0.25
 
    ## Caustic Amplitudes
    #phi20 = ax**2/rF2 + lc*psi20
    #phi02 = ay**2/rF2 + lc*psi02
    #phi11 = lc*psi11
    #phi30 = lc*V30
    #phi21 = lc*V21
    #phi12 = lc*V12
    #phi03 = lc*V03
    #B = phi20**3*phi03 - 3*phi20**2*phi11*phi12 + 3*phi20*phi11**2*phi21 - phi11**3*phi30
    #ampcaus = ax*ay/(2*pi*rF2) * 2.**(5./6.) * pi**(1./2.) * gfunc(1./3.) * \
    #        np.abs(phi20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.))
    #field = ampcaus*np.exp(1j*(phase + pshift))
    #dynspec = np.real(np.multiply(field, np.conj(field)))
    
    ###########################################################################
    
    upx = ux + alp*scrgx(ux, uy, theta, phi, N, sigma)/ax**2
    upy = uy + alp*scrgy(ux, uy, theta, phi, N, sigma)/ay**2
    rays = np.array([upx, upy])
    
    # Calculate Amplitude, Field, Phase and Phase Shift
    alp = rF2*lc
    psi20 = scrgxx(upx, upy, theta, phi, N, sigma)
    psi02 = scrgyy(upx, upy, theta, phi, N, sigma)
    psi11 = scrgxy(upx, upy, theta, phi, N, sigma)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    H = phi20*phi02 - phi11**2
    sigma = np.sign(phi02)
    delta = np.sign(H)
    amp = (ax*ay/rF2)*np.abs(H)**-0.5
    
    phase = 0.5*rF2*lc**2*((scrgx(upx, upy, theta, phi, N, sigma)/ax)**2 + \
                        (scrgy(upx, upy, theta, phi, N, sigma)/ay)**2) + \
                        lc*scrfun(upx, upy, theta, phi, N, sigma) - 0.5*pi
    
    pshift = pi*(delta + 1)*sigma*0.25
    
    # Caustic Amplitudes
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    phi30 = lc*scrgxxx(upx, upy, theta, phi, N, sigma)
    phi21 = lc*scrgxxy(upx, upy, theta, phi, N, sigma)
    phi12 = lc*scrgxyy(upx, upy, theta, phi, N, sigma)
    phi03 = lc*scrgyyy(upx, upy, theta, phi, N, sigma)
    B = phi20**3*phi03 - 3*phi20**2*phi11*phi12 + 3*phi20*phi11**2*phi21 - phi11**3*phi30
    ampcaus = ax*ay/(2*pi*rF2) * 2.**(5./6.) * pi**(1./2.) * gfunc(1./3.) * \
            np.abs(phi20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.))
    
    field = ampcaus*np.exp(1j*(phase + pshift))
    dynspec = np.real(np.multiply(field, np.conj(field)))

    ############################################################################

    #grad = lensg(ux, uy)
    #upx = ux + alp*grad[0]/ax**2
    #upy = uy + alp*grad[1]/ay**2
    #rays = np.array([upx, upy])
#
    ## Calculate Amplitude, Field, Phase and Phase Shift
    #alp = rF2*lc
    #psi20, psi02, psi11 = lensh(ux, uy)
    #phi20 = ax**2/rF2 + lc*psi20
    #phi02 = ay**2/rF2 + lc*psi02
    #phi11 = lc*psi11
    #H = phi20*phi02 - phi11**2
    #sigma = np.sign(phi02)
    #delta = np.sign(H)
    #amp = (ax*ay/rF2)*np.abs(H)**-0.5
#
    #grad = lensg(ux, uy)
    #phase = 0.5*rF2*lc**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + lc*lensfun(*uvec) - 0.5*pi
#
    #pshift = pi*(delta + 1)*sigma*0.25
#
    ## Caustic Amplitudes
    #phi20 = ax**2/rF2 + lc*psi20
    #phi02 = ay**2/rF2 + lc*psi02
    #phi11 = lc*psi11
    #phi30, phi21, phi12, phi03 = lc*np.asarray(lensgh(ux, uy))
    #B = phi20**3*phi03 - 3*phi20**2*phi11*phi12 + 3*phi20*phi11**2*phi21 - phi11**3*phi30
    #ampcaus = ax*ay/(2*pi*rF2) * 2.**(5./6.) * pi**(1./2.) * gfunc(1./3.) * \
    #        np.abs(phi20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.))
#
    #field = ampcaus*np.exp(1j*(phase + pshift))
    #dynspec = np.real(np.multiply(field, np.conj(field)))

    return rays, ampcaus, phase, field, dynspec, pshift


def findIntersection(p1, p2):
    v1 = p1.vertices
    v2 = p2.vertices
    poly1 = geometry.LineString(v1)
    poly2 = geometry.LineString(v2)
    intersection = poly1.intersection(poly2)
    # print(intersection)
    try:
        coo = np.ones([100, 2])*1000
        for a in range(len(intersection)):
            coo[a] = np.asarray(list(intersection[a].coords))
    except:
        try:
            coo = np.asarray(list(intersection.coords))
        except:
            pass
    coo = coo[np.nonzero(coo - 1000)]
    return coo

def findRoots(rays, rx, ry):
    raysx = rays[0]
    raysy = rays[1]
    cs0 = plt.contour(rx, ry, raysx, levels = [0, np.inf], colors = 'red')
    cs1 = plt.contour(rx, ry, raysy, levels = [0, np.inf], colors = 'blue')
    c0 = cs0.collections[0]
    c1 = cs1.collections[0]
    paths0 = c0.get_paths()
    paths1 = c1.get_paths()
    roots = np.array([])
    for p0 in paths0:
        for p1 in paths1:
            root = findIntersection(p0, p1)
            if len(root) != 0:
                roots = np.append(roots, root)
    roots = np.asarray(roots).flatten().reshape(-1, 2)
    p = np.argsort(roots.T[0])
    roots_all = roots[p]
    return roots_all