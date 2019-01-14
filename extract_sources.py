from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from photutils import aperture_photometry, Background2D, MedianBackground, CircularAnnulus, CircularAperture
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os, sys, warnings

# Sources to parse in each image
src_list = [(390, 348), (422, 627)]

# Parameters
sclip        = 3.0  # Sigma clipping
bg_wsize     = 50   # Pixel number for background filtering
aperture_r   = 3.0  # Radius of the aperture
sky_in       = 6.0  # Radius of inner sky-annulus
sky_out      = 8.0  # Radius of outer sky-annulus
mod_fit_size = 7    # Moffat fitting windows
plot_fits    = True # Plotting the 3D surface + fitted center ? Required a folder fits/

# For the animation part, wid indicates which source from src_list we want to plot
wid       = 1
windows   = []
xvv       = None
yvv       = None

def extract_photometry(fn):
    f = fits.open(fn)
    d1 = f[1].data

    # Background subtraction : https://photutils.readthedocs.io/en/stable/background.html
    sigma_clip = SigmaClip(sigma=sclip)
    bkg_estimator = MedianBackground()
    bkg = Background2D(d1, (bg_wsize, bg_wsize), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    d2 = d1 - bkg.background

    # Building local positions, fitting Moffat2D models
    positions = []
    invalid   = []
    
    for i, pt in enumerate(src_list):
        # Extracting the window for fitting
        x, y = pt
        x_min  = x - mod_fit_size
        x_max  = x + mod_fit_size
        y_min  = y - mod_fit_size
        y_max  = y + mod_fit_size
        window = d2[y_min:y_max+1, x_min:x_max+1]

        # Initial guess 
        z0     = d2[y, x]
        m_init = models.Moffat2D(z0, x, y)

        # Fitting, we catch warnings as exceptions in case the fit fails
        with warnings.catch_warnings(record=True) as w:
            fit_m  = fitting.LevMarLSQFitter()
            xv, yv = np.meshgrid(range(x_min, x_max+1), range(y_min, y_max+1))
            p      = fit_m(m_init, xv, yv, window)

            if w and issubclass(w[-1].category, AstropyUserWarning):
                print('Warning : The fit might not have converged for source #{} at position {}'.format(i, pt))
                invalid.append(i)
        
        px = p.x_0.value
        py = p.y_0.value
        pz = p.amplitude.value

        # Storing info for animation
        if i==wid:
            global xvv, yvv, windows
            xvv = xv
            yvv = yv

            ix = int(round(px))
            iy = int(round(py))

            x_min  = ix - mod_fit_size
            x_max  = ix + mod_fit_size
            y_min  = iy - mod_fit_size
            y_max  = iy + mod_fit_size

            nw = d2[y_min:y_max+1, x_min:x_max+1]
            windows += [nw]

        # Rendering fit to file
        if os.path.exists('fits') and plot_fits:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xv, yv, window)
            ax.scatter(px, py, pz, s=3, color='red')
            plt.savefig('fits/fit_{}_{}_{}.png'.format(fn, x, y))
            plt.close('all')

        positions += [(p.x_0.value, p.y_0.value)]

    # Aperture photometry : https://photutils.readthedocs.io/en/stable/aperture.html
    apertures = CircularAperture(positions, r=aperture_r)
    annulus_apertures = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)
    apers = [apertures, annulus_apertures]

    phot_table = aperture_photometry(d2, apers)

    # Mean sky subtraction
    bkg_mean  = phot_table['aperture_sum_1'] / annulus_apertures.area()
    bkg_sum   = bkg_mean * apertures.area()
    final_sum = phot_table['aperture_sum_0'] - bkg_sum

    # Calculating zero-point : http://www.stsci.edu/hst/wfpc2/analysis/wfpc2_cookbook.html
    h0 = f[0].header
    h1 = f[1].header

    phot_zpt  = h1['PHOTZPT']
    phot_flam = h1['PHOTFLAM']
    zero_pt = -2.5 * np.log10(phot_flam) + phot_zpt

    # TODO : Correct from STMAG to Cousins
    magnitudes = []
    for i, flux in enumerate(final_sum):
        if i in invalid:
            magnitudes.append(np.nan)
        else:
            m = -2.5 * np.log10(flux) + zero_pt
            magnitudes.append(m)
        
    print(fn, h0['EXPEND'], magnitudes)
    return h0['EXPEND'], magnitudes

def extract_current_folder():
    ''' 
    Extracts all photometry in the current folder, and animates one of the result sources
    Only treats fits file starting with hst and finishing with .fits
    '''
    global windows, xvv, yvv
    print('Extracting sources at positions : ', src_list)
    
    print('Filename\tEpoch\tMagnitudes')
    filenames = os.listdir('.')
    filenames.sort()
    mags  = []
    times = []
    for f in filenames:
        if not f.startswith('hst') or not f.endswith('.fits'):
            continue

        # Extracting time and magnitude from current file
        time, mag = extract_photometry(f)
        times.append(time)
        mags.append(mag)

    mags  = np.asarray(mags)
    times = np.asarray(times)

    # Plotting the magnitudes 1 by 1 to remove the NaNs
    for sid, _ in enumerate(src_list):
        # Sorting the files and magnitudes in Epoch order in case the files are
        # not already sorted
        mask = ~np.isnan(mags[:,sid])
        
        cur_mags  = mags[mask,sid]
        cur_times = times[mask]

        N = cur_mags.shape[0]

        sorted_ids = [i for i in range(N)]
        sorted_ids.sort(key=lambda x:cur_times[x])
        
        cur_mags  = cur_mags[sorted_ids]
        cur_times = cur_times[sorted_ids]
        plt.plot(cur_times, cur_mags, '-+')
    plt.show()


    # Animation part ... A bit ugly
    if wid > -1:
        # Ordering times
        N = len(windows)
        sorted_ids = [i for i in range(N)]
        sorted_ids.sort(key=lambda x:times[x])
        
        # 3D matrix : dim 0 -> Time, dim 1 -> y, dim 2 -> x
        windows = np.asarray(windows)[sorted_ids, :, :]
    
        max_v = windows.max()
        min_v = windows.min()
        z     = windows[0]

        # Building figure
        fig   = plt.figure(figsize=(10, 10))
        ax    = fig.add_subplot(111, projection='3d')
        p     = ax.plot_surface(xvv, yvv, z)

        # Number of interpolation points
        inter_t = 60
        inter_x = 100
        inter_y = 100
        
        ox = xvv[0,:]
        oy = yvv[:,0]
        ot = times[sorted_ids]
        

        # Generating vector coords and meshgrid
        tv = np.linspace(ot.min(), ot.max(), inter_t)
        xv = np.linspace(xvv.min(), xvv.max(), inter_x)
        yv = np.linspace(yvv.min(), yvv.max(), inter_y)
        MT, MX, MY = np.meshgrid(tv, xv, yv, indexing='ij')
        
        # Rolling back to 1D for interpolation
        MT = MT.ravel()
        MX = MX.ravel()
        MY = MY.ravel()

        # Interpolation
        pos = np.stack((MT, MX, MY)).T
        f = RegularGridInterpolator((ot, ox, oy), windows)
        w_array = f(pos)
        w_array = w_array.reshape((inter_t, inter_x, inter_y))

        # Rebuilding 2D position matrices
        MX, MY = np.meshgrid(xv, yv)

        # Update function for matplotlib's anim
        def update(i):
            print('Rendering frame {}/{}'.format(i+1, inter_t))
            z = w_array[i]
            ax.clear()
            p = ax.plot_surface(MX, MY, z)
            ax.set_zlim(min_v, max_v)
            return p,

        def init():
            update(0)

        # And rendering 
        ani = FuncAnimation(fig, update, init_func=init, frames=range(1, inter_t))
        Writer = animation.writers['ffmpeg']
        w = Writer(fps=15, bitrate=1800)
        print('Saving animation')
        ani.save('pulse.mp4', writer=w)
        plt.close('all')



    
if __name__ == '__main__':
    extract_current_folder()
