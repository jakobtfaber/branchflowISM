
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

from RayTrace3D_Thesis_Utils import *

# We are going to solve the lens equation to find the mapping between the
# u' and u plane numerically using a root finding algorithm.
# For now, however, this won't be necessary - but we'll set up the tools for later (3.2.21).
# Compute 100 zeros and values of the Airy function Ai and its derivative
# where index 1 for ai_zeros() returns the first 100 zeros of Ai’(x) and
# index 0 for airy() returns Ai'(x).
# Note for later: A caustic in which two images merge corresponds to a
# fold catastrophe, and close to a fold the field follows an Airy function pattern.
airyzeros = ai_zeros(100)[1]
airyfunc = airy(airyzeros)[0]**2/2.
airsqrenv = interp1d(airyzeros, airyfunc, kind = 'cubic', fill_value = 'extrapolate')

# Define screen paramters (Gaussian & Kolmogorov)
nscreen = 4
N = 10
# Define screen paramters (Gaussian & Kolmogorov)
sigma = 20
iscreen = 1 #which wavefront to plot
dso = 1*kpc*pctocm #distance from source to observer
dsl = 1*kpc*pctocm / nscreen #distance from source to screen
dm = 1e-5*pctocm #dispersion measure
ax, ay = 0.08*autocm, 0.08*autocm #screen width (x,y)
uxmax, uymax = 25., 25. #screen coordinates

# Frequencies
fmin, fmax = 1.*GHz, 1.1*GHz #min/max frequency
nchan = 1
freqs = np.linspace(fmin, fmax, nchan)
#print('Observation Frequency (GHz): ', (fmax+fmin)//2 * 1e-9)

# Construct u plane
npoints = 150
rx = np.linspace(-uxmax, uxmax, npoints)
ry = np.linspace(-uymax, uymax, npoints)
#rx = np.linspace(-1, 1, npoints)
#ry = np.linspace(-1, 1, npoints)
uvec = np.meshgrid(rx, ry)
ux, uy = uvec


raypropsteps = np.zeros((nchan, nscreen, 2, npoints, npoints)) #store ray wavefront at each screen
screens = np.zeros((nscreen, npoints, npoints)) #store phases at each screen
phases = np.zeros((nchan, nscreen, npoints, npoints))
phaseshift = np.zeros((nchan, nscreen, npoints, npoints))
fields = np.zeros((nchan, nscreen, npoints, npoints), dtype = complex)
dynspecs = np.zeros((nchan, nscreen, npoints, npoints), dtype = complex)
amps = np.zeros((nchan, nscreen, npoints, npoints))
thetas = np.zeros((nscreen, N))
phis = np.zeros((nscreen, N))
roots = {}

for scr in range(nscreen):

    thetas[scr] = np.random.uniform(0,2*np.pi, N)
    phis[scr] = np.random.uniform(0, 2*np.pi, N)
    screens[scr] = scrfun(ux, uy, thetas[scr], phis[scr], N, sigma)
    V = np.zeros((npoints, npoints))
    rF2 = rFsqr(dso, dsl, freqs[0])
    rF = np.sqrt(rF2)
    #print(rF)
    AR = np.random.randint(1, 30)
    #print(AR)
    #V = Kolmogorov(npoints, npoints, dx = 0.05*rF, dy = 0.05*rF, rf = rF, psi = 0, ar = 4, alpha = 5./3., mb2 = 20., inner = 0.001*rF)
    #screens[scr] = V

for n in range(nchan):

    print('Channel Frequency: ', freqs[n])

    # Calculate coefficients for the scr equation
    rF2 = rFsqr(dso, dsl, freqs[n])#freqs[n])
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])
    lc = lensc(dm, freqs[n]) #freqs[n]) #calculate phase perturbation due to the scr
    #print('Phase Pertrubation $\phi_{0}$: ', lc)
    alp  = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])

    for scr in range(nscreen):

        if scr == 0:
            #map_ = mapToUprime(uvec, alp, ax, ay, rF2, lc, sigma, thetas[scr], phis[scr], N, screens[scr])
            map_ = mapToUprime(uvec, alp, ax, ay, rF2, lc, sigma, thetas[scr], phis[scr], N)
            raypropsteps[n][scr] = map_[0]
            amps[n][scr] = map_[1]
            phases[n][scr] = map_[2]
            fields[n][scr] = map_[3]
            dynspecs[n][scr] = map_[4]
            phaseshift[n][scr] = map_[5]
            roots[n] = {}
            roots[n][scr] = findRoots(map_[0], rx, ry)
            screens[scr] = screens[scr]

            
        else:
            #map_ = mapToUprime(raypropsteps[n][scr-1], alp, ax, ay, rF2, lc, sigma, thetas[scr], phis[scr], N, screens[scr])
            map_ = mapToUprime(raypropsteps[n][scr-1], alp, ax, ay, rF2, lc, sigma, thetas[scr], phis[scr], N)
            raypropsteps[n][scr] = map_[0]
            amps[n][scr] = map_[1]
            phases[n][scr] = map_[2]
            fields[n][scr] = map_[3]
            dynspecs[n][scr] = map_[4]
            phaseshift[n][scr] = map_[5]
            roots[n] = {}
            roots[n][scr] = findRoots(map_[0], rx, ry)
            screens[scr] = screens[scr]

chan = 0
print('Plotted Frequency (GHz): ', freqs[chan] * 1e-9)
screen = 3
print('Plotted Screen: ', screen)


# Construct Dynamic Spectrum
dynspec = np.zeros((nscreen, nchan, npoints))

for s in range(nscreen):
    for n in range(nchan):
        dynspec[s][n] = dynspecs[n][s][npoints//2, :]

# Construct Secondary Spectrum

secfft = np.fft.fftn((dynspec[screen])-np.mean(dynspec[screen]))
secreal = np.absolute(np.fft.fftshift(secfft))**2
secspec = 10*np.log10(secreal/np.max(secreal))

#secfft = np.fft.fftn((dynspecs[chan][screen])-np.mean(dynspecs[chan][screen]))
#secreal = np.absolute(np.fft.fftshift(secfft))**2
#secspec = 10*np.log10(secreal/np.max(secreal))

fig = plt.figure(figsize = (20, 10))

ax0 = fig.add_subplot(231)
plt.imshow(screens[screen], aspect = 'auto')
plt.title('Screen')

ax1 = fig.add_subplot(232)

allphases = np.zeros((npoints, npoints))
for s in range(nscreen):
    allphases += phases[chan][screen]
#plt.imshow(allphases, aspect = 'auto')

secfftp = np.fft.fftn((allphases)-np.mean(allphases))
secrealp = np.absolute(np.fft.fftshift(secfftp))**2
secspecp = 10*np.log10(secrealp/np.max(secrealp))


#plt.plot(secspecp[npoints//2+0:npoints//2+5, :].sum(0))#, aspect = 'auto')
#plt.plot(secspecp[npoints//2+5:npoints//2+10, :].sum(0)+50)#, aspect = 'auto')
#plt.plot(secspecp[npoints//2+10:npoints//2+15, :].sum(0)+100)#, aspect = 'auto')
#plt.plot(secspecp[npoints//2+15:npoints//2+20, :].sum(0)+150)#, aspect = 'auto')
#plt.plot(secspecp[npoints//2+20:npoints//2+25, :].sum(0)+200)#, aspect = 'auto')
#plt.imshow(secspecp, aspect = 'auto', vmin = -45, vmax = 0)
plt.imshow(allphases, aspect = 'auto')
plt.colorbar()
print('Max: ', np.max(secspecp))
plt.title('Phases')

ax3 = fig.add_subplot(233)

plt.imshow(phaseshift[chan][screen], aspect = 'auto')
plt.title('Phase Shifts')

ax4 = fig.add_subplot(234)

xflat = np.ndarray.flatten(np.array(raypropsteps[chan][screen][0]))
yflat = np.ndarray.flatten(np.array(raypropsteps[chan][screen][1]))
plt.scatter(xflat, yflat, c = 'k', s = 0.02)
plt.title('Ray Tracing')

ax5 = fig.add_subplot(235)

plt.hist2d(xflat, yflat, bins = npoints//2)
plt.title('Ray Density')

ax6 = fig.add_subplot(236)

plt.imshow(amps[chan][screen], aspect = 'auto')
plt.title('Amplitude')

plt.tight_layout()
plt.show()
fig.savefig('Diagnostic1_Sig' + str(sigma) + '.png')

fig1 = plt.figure(figsize = (20, 10))

ax2 = fig1.add_subplot(231)

plt.imshow(fields[chan][screen].real, aspect = 'auto')
plt.title('Electric Field')

ax3 = fig1.add_subplot(232)

plt.plot(amps[chan][screen][:, npoints//2])
plt.title('Amplitude (Central Slice)')

ax4 = fig1.add_subplot(233)

plt.plot(fields[chan][screen][:, npoints//2])
plt.title('Electric Field (Central Slice)')

ax5 = fig1.add_subplot(234)

plt.imshow(dynspec[screen], aspect = 'auto')
plt.ylabel('Frequency: ' + str(round(fmin*1e-9, 2)) + '-' + str(round(fmax*1e-9, 2)) + ' GHz')
plt.title('Dynamic Spectrum')

ax6 = fig1.add_subplot(235)
# get electric field impulse response
#p = np.fft.fft(np.multiply(dynspec[screen], np.blackman(nchan)[:, None]), 2*nchan)
#p = np.real(p*np.conj(p))  # get intensity impulse response
# shift impulse to middle of window
#pulsewin = np.transpose(np.roll(p, nchan))
#Freq = freqs/1000
#lpw = np.log10(pulsewin)
#vmax = np.max(lpw)
#vmin = np.median(lpw) - 3
#plt.imshow(lpw, aspect = 'auto', vmin = vmin, vmax = vmax)
#plt.pcolormesh(np.linspace(0, uxmax, nchan),
#              (np.arange(0, nchan, 1) - nchan/2) /
#               (2*(c/nchan)*Freq),
#               lpw[int(nchan/2):, :], vmin=vmin, vmax=vmax)
#plt.colorbar
#plt.ylabel('Delay (ns)')
#plt.xlabel('$x/r_f$')

#plt.plot(np.linspace(0, nchan, nchan),
#         -dm/(2*(c/nchan)*Freq), 'k')  # group delay=-phase delay
for i in range(nchan):
    plt.scatter(roots[i][scr].T[0], roots[i][scr].T[1], color = 'black')
    
ax7 = fig1.add_subplot(236)

plt.imshow(secspec.T, aspect = 'auto')
plt.title('Secondary Spectrum')
    
plt.tight_layout()
plt.show()
fig1.savefig('Diagnostic2_Sig' + str(sigma) + '.png')


x_dat = raypropsteps[:, :, 0, :, :]
y_dat = raypropsteps[:, :, 1, :, :]

propx = np.zeros((nscreen, (npoints*npoints)))
propy = np.zeros((nscreen, (npoints*npoints)))

for s in range(nscreen):
    propx[s] = np.ndarray.flatten(x_dat[chan][s])
    propy[s] = np.ndarray.flatten(y_dat[chan][s])


# Plot wavefront scatter plots (both singular and 4x4)

fig1 = plt.figure(figsize=(10,10))
fig1.tight_layout()
ax1 = plt.subplot(111)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

scat = ax1.scatter(propx[0], propy[0], c = 'k', s=0.05, \
            label = r'$\alpha$ = ' + '5/3' + '\n' + r'$\nu = $' + '{:.2e}'.format(fmax) + \
            ' Hz' + '\n' + 'dso = 2 kpc' + '\n' + 'dscr = dso /' + str(nscreen))
ax1.set_title('Transverse Ray Paths ($N_{scr} = $' + str(nscreen) + ')')
ax1.set_xlabel('$u^\prime_{x} (2 AU)$')
ax1.set_ylabel('$u^\prime_{y} (2 AU)$')
ax1.set_xlim(-7, 7)
ax1.set_ylim(-7, 7)
ax1.legend(loc = 'upper right')

# Animation update function
def animationUpdate(k):
    x = propx[k]
    y = propy[k]
    scat.set_offsets(np.c_[x,y])
    return scat,

# function for creating animation
anim = FuncAnimation(fig1, animationUpdate, frames=int(nscreen-1), interval=200) #, blit=True)#, repeat_delay=3000)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer()
#anim.save('RayPathsTransverse_Video.mp4', writer=writer, dpi=300)

fig2 = plt.figure(figsize = (20, 10))
fig2.tight_layout()
ax = fig2.add_subplot(221)

s = np.linspace(0, nscreen, nscreen)
for i in range(npoints):
    plt.plot(s, x_dat[chan, :, npoints//2, i], c = 'k', linewidth = 0.05)
plt.title('Longitudinal Ray Paths ($y = 0$)')
plt.ylabel('$x$ (2 AU)')
plt.xlabel('N$_{scr}$ ($z$; l$_{total}$ = 2 kpc)')
    
ax2 = fig2.add_subplot(222)
s = np.linspace(0, nscreen, nscreen)
for i in range(npoints):
    plt.plot(s, y_dat[chan, :, i, npoints//2], c = 'k', linewidth = 0.05)
plt.title('Longitudinal Ray Paths ($x = 0$)')
plt.ylabel('$y$ (2 AU)')
plt.xlabel('N$_{scr}$ ($z$; l$_{total}$ = 2 kpc)')
    
ax3 = fig2.add_subplot(223)
s = np.linspace(0, nscreen, nscreen)
for i in range(npoints):
    plt.plot(s, x_dat[chan, :, i, 0] - np.mean(x_dat[chan, :, 0, 0]), c = 'k', linewidth = 0.05)
plt.ylabel('Momentum ($p_{x}$)')
plt.xlabel('N$_{scr}$ ($z$; l$_{total}$ = 2 kpc)')

    
ax4 = fig2.add_subplot(224)
s = np.linspace(0, nscreen, nscreen)
for i in range(npoints):
    plt.plot(s, y_dat[chan, :, 0, i] - np.mean(y_dat[chan, :, 0, 0]), c = 'k', linewidth = 0.05)
plt.ylabel('Momentum ($p_{y}$)')
plt.xlabel('N$_{scr}$ ($z$; l$_{total}$ = 2 kpc)')

plt.show()
#fig2.savefig('RayPathsLongitudinal_4x4.png')


fig = plt.figure(figsize = (25, 10))
ax = fig.add_subplot(211)
plt.plot(x_dat[chan, :, npoints//2, :], c = 'k')
xdat = x_dat[chan, :, :, :]
var = np.zeros(nscreen)
for s in range(nscreen):
    var[s] = np.var(xdat[s])
ax1 = fig.add_subplot(212)
plt.plot(var)
plt.show()