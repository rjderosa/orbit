import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import ascii
import emcee
import scipy.stats as stats
import scipy.special as special
import time
from astropy.io import fits
import corner
import multiprocessing as mp
import itertools

    # 0 log(a) - Semi-major axis (asec)
    # 1 cos(i) - Inclination (rad)
    # 2 e - Eccentricity
    # 3 w - Argument of periapsis or (Omega + omega)
    # 4 o - Longitude of ascending node or (Omega - omega)
    # 5 tau - Periastron passage in fractions of period past epoch[0]
    # 6 plx - Parallax to star (mas)
    # 7 Mtotal - System mass limited to <= 50 Msol
    # (8 - mass ratio)
    # (9 - rest velocity)
    # epcohs - array of MJDs


def lnlike(param, epochs, rx, rx_err, ty, ty_err, va_obs, va_obs_err, rho_theta, rv_a, sumdiff):

    #Reference T0 = epochs[0]
    #So tau varies between 0P and 1P
    #rx - either rho or x
    #ty - either theta or y
    
    a = 10.0**param[0]
    P = np.sqrt(((a * (1000.0/param[6]))**3.0)/param[7])
    T0 = (param[5]*(P*365.25))+epochs[0]

    if rv_a is True:
        q, vrest = param[8], param[9]
    else:
        q, vrest = 1.0, 0.0

    result = elements_to_xy(P, a, np.arccos(param[1]), param[2], param[3], param[4], T0, epochs, rtheta = True, rvs = rv_a, plx = param[6], q = q, vrest = vrest, sumdiff = sumdiff)
    x, y, r, t, va, vb = result

    if rho_theta is True:
        d_theta = np.arctan2(np.sin(ty - t), np.cos(ty - t))
        chi2 = np.sum(((rx - r)/rx_err)**2.0)
        chi2 += np.sum((d_theta/ty_err)**2.0)
    else:
        chi2 = np.sum(((rx - x)/rx_err)**2.0)
        chi2 += np.sum(((ty - y)/ty_err)**2.0)

    if rv_a is True:
        chi2 += np.sum(((va_obs - va)/va_obs_err)**2.0)

    return -0.5*chi2


def lnprior(param, epoch0, star_plx, star_mass, e_planet, u_limits, sumdiff):

    #Limit to distance > 0.5 pc, < 1000 pc, Mass < 50 Msol
    if (-1.0 <= param[1] <= 1.0) & \
        (0.0 <= param[2] <  1.0) & \
        (0.0 <= param[3] <  (2.0*np.pi)) & \
        (0.0 <= param[4] <  (2.0*np.pi)) & \
        (0.0 <= param[5] <  1.0) & \
        (1.0 <= param[6] <= 2000.0) & \
        (0.0 <  param[7] <= 50.0):

        #Only if fitting RV
        if len(param) > 8:
            if ((param[8] <= 0.0) | (param[8] >= 5.0)):
                return -np.inf

        a = 10.0**param[0]
        P = np.sqrt(((a * (1000.0/param[6]))**3.0)/param[7])

        #Check upper limits
        if u_limits is not False:
            T0 = (param[5]*(P*365.25))+epoch0
            x, y, r, t, va, vb = elements_to_xy(P, a, np.arccos(param[1]), param[2], param[3], param[4], T0, u_limits[:,0], rtheta = True, rvs = False, plx = param[6], sumdiff = sumdiff)
            if True in (r > u_limits[:,1]):
                return -np.inf

        if (P > 0.0027):
            if e_planet is True:
                #Updated to use the beta prior for long-period planets from Kipping 2013
                #1/B(a,b) e^a-1 (1-e)^b-1
                p_e = (1.0/special.beta(1.12, 3.09))*(param[2]**0.12)*((1.0-param[2])**2.09)
                p_e = -np.inf if p_e <= 0.0 else np.log(p_e)
            else:
                p_e = 0.0

            p_plx = (1.0/((param[6]/1000.0)**4.0)) / 2.66667e9 #Constant space density
            if star_plx[0] != 0.0:
                p_plx *= np.exp(-(((param[6]-star_plx[0])**2)/(2.0*star_plx[1]*star_plx[1])))
            p_plx = -np.inf if p_plx <= 0.0 else np.log(p_plx)

            if star_mass[0] == 0.0:
                p_mass = 0.0
            else:
                p_mass = np.exp(-(((param[7]-star_mass[0])**2)/(2.0*star_mass[1]*star_mass[1])))
                p_mass = -np.inf if p_mass <= 0.0 else np.log(p_mass)
                
            return p_e + p_plx + p_mass
        else:
            return -np.inf
    else:
        return -np.inf

def run_emcee(data_file, star_plx, star_mass, mjd = True, rho_theta = True, rv_a = False, e_planet = False, guess = False, pos0 = False, u_limits = False, ntemps = 16, nwalkers = 1000, nsteps = 1000, nrepeats = 100, nburn = 0, nthreads = 20, ndec = 100, benchmark = False, sumdiff = False):

    #ndec = decimation - decimate every ndec in sampler array. nsteps and ndec should be divisible

    dr = (np.pi / 180.0)
    rd = (180.0 / np.pi)

    if (sumdiff is True) and (rv_a is True):
        print 'Cannot use Omega + omega, Omega - omega with RV data\nSetting sumdiff to False'
        sumdiff = False

    data = read_data(data_file, mjd = mjd, rho_theta = rho_theta, rv_a = rv_a)
    epochs = data[0]

    if rho_theta is True:
        rho = data[1]
        rho_err = data[2]
        theta = data[3]
        theta_err = data[4]
    else:
        dx = data[1]
        dx_err = data[2]
        dy = data[3]
        dy_err = data[4]

    if rv_a is True:
        va_obs = data[5]
        va_obs_err = data[6]
    else:
        va_obs = np.copy(epochs) * np.nan
        va_obs_err = np.copy(epochs) * np.nan

    n = len(epochs)

    if pos0 is False:
        #Now to intialize the starting array
        multiplier = -1
        n_okay = 0
        while n_okay < (ntemps * nwalkers):
            multiplier += 2
            if rv_a is True:
                ndim = 10
                pos0 = np.zeros((ntemps * nwalkers * multiplier, ndim), dtype = np.float64)
                if guess is False:
                    p0 = np.array([0.0, 0.0, 0.50, np.pi, np.pi, 0.50, star_plx[0], star_mass[0], 0.5, 0.0],dtype=np.float64)
                    dp = np.array([1.0, 1.0, 0.50, np.pi, np.pi, 0.50, star_plx[1], star_mass[1], 0.25, 1.0], dtype=np.float64)
                else:
                    p0 = np.array(guess).astype(np.float64)
                    dp = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, star_plx[1], star_mass[1], 0.1, 0.1], dtype=np.float64)
                #Set limits on parameters
                p_min = [-100,  -1.0,  0.0,         0.0,         0.0,   0.0,    1.0, 0.01, 0.0001, -1000000]
                p_max = [ None,  1.0, 0.99, 1.999*np.pi, 1.999*np.pi, 0.999, 2000.0, 50.0,   None,  1000000]

            else:
                ndim = 8
                pos0 = np.zeros((ntemps * nwalkers * multiplier, ndim), dtype = np.float64)
                if guess is False:
                    p0 = np.array([0.0, 0.0, 0.50, np.pi, np.pi, 0.50, star_plx[0], star_mass[0]],dtype=np.float64)
                    dp = np.array([1.0, 1.0, 0.50, np.pi, np.pi, 0.50, star_plx[1], star_mass[1]], dtype=np.float64)
                else:
                    p0 = np.array(guess).astype(np.float64)
                    dp = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, star_plx[1], star_mass[1]], dtype=np.float64)
                #Set limits on parameters
                p_min = [-100,  -1.0,  0.0,         0.0,         0.0,   0.0,    1.0, 0.01]
                p_max = [ None,  1.0, 0.99, 1.999*np.pi, 1.999*np.pi, 0.999, 2000.0, 50.0]

            #If we aren't constraining the mass there needs to be a sensible estimate
            if star_mass[0] == 0.0:
                p0[7] = 1.0
                dp[7] = 0.1

            if star_plx[0] == 0.0:
                p0[6] = 10.0
                dp[6] = 1.0

            for i in xrange(0, ndim):
                #Because of angles, randomly sample half of the two omegas, and move them to (x + 180 MOD 360)
                #This shouldn't do anything for the wide range sample
                #Only do this if we are using a guess and are not using RVs (as RV breaks this degeneracy)
                #If we don't have a guess, the angles will be distributed between 0->2 pi anyway
                #Does not need to be done if using (O+o) (O-o)
                if (((i == 3) | (i == 4)) & (guess is not False) & (rv_a is False) & (sumdiff is False)):
                    rand_p = np.random.uniform(low=p0[i]-dp[i], high=p0[i]+dp[i], size = (ntemps * nwalkers * multiplier)).clip(min = p_min[i], max = p_max[i])
                    rand_u = np.random.uniform(low = -1, high = 1, size = (ntemps * nwalkers * multiplier))
                    ind = np.where(rand_u < 0.0)
                    rand_p[ind] = ((rand_p[ind] + np.pi) % (2.0*np.pi))
                    pos0[:,i] = rand_p    
                else:
                    pos0[:,i] = np.random.uniform(low=p0[i]-dp[i], high=p0[i]+dp[i], size = (ntemps * nwalkers * multiplier)).clip(min = p_min[i], max = p_max[i])

            #Need to do remove all those that don't satisfy the upper limit prior
            if u_limits is not False:
                a = 10.0**pos0[:,0]
                P = np.sqrt(((a * (1000.0/pos0[:,6]))**3.0)/pos0[:,7])
                T0 = (pos0[:,5]*(P*365.25))+epochs[0]

                okay = np.zeros((ntemps * nwalkers * multiplier), dtype=int)
                okay[:] = 1
                for i in xrange(0, ntemps * nwalkers * multiplier):
                    x, y, r, t, va, vb = elements_to_xy(P[i], a[i], np.arccos(pos0[i,1]), pos0[i,2], pos0[i,3], pos0[i,4], T0[i], u_limits[:,0], rtheta = True, rvs = False, plx = pos0[i,6], sumdiff = sumdiff)
                    if True in (r > u_limits[:,1]):
                        okay[i] = 0

                n_okay = np.sum(okay)
                ind = np.where(okay == 1)[0]
                if multiplier == 1:
                    print 'Generating acceptable starting positions'
                print np.shape(ind), np.shape(ind[0:ntemps*nwalkers])
                if n_okay >= (ntemps * nwalkers):
                    pos0 = pos0[ind[0:ntemps*nwalkers],:]

            else:
                n_okay = ntemps * nwalkers #To escape from while

        pos0 = np.reshape(pos0, (ntemps, nwalkers, ndim))
        #fits.writeto('pos0.fits', pos0, overwrite=True)
    else:
        ndim = np.shape(pos0)[2]

    if benchmark is True:
        #Benchmark
        ntest = 20
        total_time = np.zeros(64, dtype=np.float64)
        total_time[:] = np.nan

        for nthreads in xrange(1, mp.cpu_count(), 1):
            time0 = time.time()

            if rho_theta is True:
                loglargs = (epochs, rho, rho_err, theta, theta_err, va_obs, va_obs_err, rho_theta, rv_a, sumdiff)
            else:
                loglargs = (epochs, dx, dx_err, dy, dy_err, va_obs, va_obs_err, rho_theta, rv_a, sumdiff)

            sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, loglargs=loglargs, logpargs=(epochs[0], star_plx, star_mass, e_planet, u_limits, sumdiff), threads = nthreads)
            pos, lnp, lnl = sampler.run_mcmc(pos0, ntest)

            total_time[nthreads] = time.time()-time0
            print nthreads, total_time[nthreads]

        del sampler
        nthreads = np.nanargmin(total_time)
    
    if rho_theta is True:
        loglargs=(epochs, rho, rho_err, theta, theta_err, va_obs, va_obs_err, rho_theta, rv_a, sumdiff)
    else:
        loglargs=(epochs, dx, dx_err, dy, dy_err, va_obs, va_obs_err, rho_theta, rv_a, sumdiff)

    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, loglargs=loglargs, logpargs=(epochs[0], star_plx, star_mass, e_planet, u_limits, sumdiff), threads = nthreads)

    #Run burn in - default now is to remove burn in phase in plotting routine
    if nburn > 0:
        pos, lnp, lnl = sampler.run_mcmc(pos0, nburn)
        sampler.reset()
    else:
        pos = pos0
    
    time0 = time.time()
    for istep in xrange(0, nrepeats):
        pos, lnp, lnl = sampler.run_mcmc(pos, nsteps)

        if istep == 0:
            lo_samples = sampler.chain[0, :, ::ndec, :]
            hi_samples = sampler.chain[ntemps-1, :, ::ndec, :]
        else:
            lo_samples = np.append(lo_samples, sampler.chain[0, :, ::ndec, :], axis = 1)
            hi_samples = np.append(hi_samples, sampler.chain[ntemps-1, :, ::ndec, :], axis = 1)

        fits.writeto('Samples_nw'+str(nwalkers)+'_lo.fits', lo_samples, overwrite=True)
        fits.writeto('Samples_nw'+str(nwalkers)+'_hi.fits', hi_samples, overwrite=True)
        
        print istep, time.time()-time0, np.shape(sampler.chain), np.shape(lo_samples)

        #Now delete the sampler to save memory
        sampler.reset()

    return nwalkers

def plot_orbit(data_file, object_name, nwalkers, mjd = True, rho_theta = True, rv_a = False, u_limits = False, nburn = False , date_range = False, x_range = False, y_range = False, r_range = False, t_range = False, log_e = False, ephems = False, sumdiff = False):

    dr = (np.pi / 180.0)
    rd = (180.0 / np.pi)

    #Load data
    data = read_data(data_file, mjd = mjd, rho_theta = rho_theta, rv_a = rv_a)
    epochs = data[0]

    if (sumdiff is True) and (rv_a is True):
        print 'Cannot use Omega + omega, Omega - omega with RV data\nSetting sumdiff to False'
        sumdiff = False

    if rho_theta is True:
        rho = data[1]
        rho_err = data[2]
        theta = data[3]
        theta_err = data[4]
    else:
        dx = data[1]
        dx_err = data[2]
        dy = data[3]
        dy_err = data[4]

    if rv_a is True:
        va_obs = data[5]
        va_obs_err = data[6]
    else:
        va_obs = np.copy(epochs) * np.nan
        va_obs_err = np.copy(epochs) * np.nan
    
    n = len(epochs)

    #One extra dimension for period, one for a (au) and another for T0
    d_ndim = 0
    if rv_a is True: d_ndim += 2
    if sumdiff is True: d_ndim += 2
    ndim = 11 + d_ndim

    itemps = [0, 1]
    for itemp in itemps:
        if itemp == 0:
            samples = fits.getdata('Samples_nw'+str(nwalkers)+'_lo.fits', memmap=False)
        elif itemp == 1:
            samples = fits.getdata('Samples_nw'+str(nwalkers)+'_hi.fits', memmap=False)
        else:
            samples = 0

        if nburn is False:
            #Default to half the array
            s = np.shape(samples)
            nburn = int(np.round(s[1]/2.0))

        a = 10**samples[:,:,0]
        i = (np.arccos(samples[:,:,1]) * rd)
        e = samples[:,:,2]
        tau = (samples[:,:,5] % 1.0)
        plx = samples[:,:,6]
        M = samples[:,:,7]

        if rv_a is True:
            q = samples[:,:,8]
            vrest = samples[:,:,9]

        if sumdiff is False:
            w = (samples[:,:,3] * rd) % 360.0
            o = (samples[:,:,4] * rd) % 360.0
        else:
            #Tile all the arrays
            a = np.tile(a, (2, 1))
            i = np.tile(i, (2,1))
            e = np.tile(e, (2,1))
            tau = np.tile(tau, (2,1))
            plx = np.tile(plx, (2,1))
            M = np.tile(M, (2,1))

            plus = (samples[:, :, 3] * rd) % 360.0
            minus = (samples[:, :, 4] * rd) % 360.0
            w = (plus - minus)/2.0
            o = (plus + minus)/2.0
            w = np.vstack((w, w+180.0)) % 360.0
            o = np.vstack((o, o+180.0)) % 360.0
            plus = np.tile(plus, (2, 1))
            minus = np.tile(minus, (2, 1))

        if log_e is True:
            old_e = np.copy(e)
            e = np.log10(1.0 - e)

        a_mas = a*1000.0
        d = 1000.0/plx
        a_au = a * d
        P = np.sqrt(((a_au)**3.0)/M)
        T0 = (tau*(P*365.25))+epochs[0]
        T0_year = jd2year(T0, mjd = True)

        if itemp == 0:
            p_min = np.nanmin(P[:,nburn:])
            p_max = np.nanpercentile(P[:,nburn:], 95.0)
            a_min = np.nanmin(a_mas[:,nburn:])
            a_max = np.nanpercentile(a_mas[:,nburn:], 95.0)
            a_au_min = np.nanmin(a_au[:, nburn:])
            a_au_max = np.nanpercentile(a_au[:, nburn:], 95.0)
            T0_min = np.nanmin(T0_year[:,nburn:])
            T0_max = np.nanpercentile(T0_year[:,nburn:], 95.0)

            p_min = np.clip(p_min - ((p_max - p_min)*0.05), 0.0, None)
            a_min = np.clip(a_min - ((a_max - a_min)*0.05), 0.0, None)
            a_au_min = np.clip(a_au_min - ((a_au_max - a_au_min)*0.05), 0.0, None)
            T0_min = np.clip(T0_min - ((T0_max-T0_min)*0.05), 0.0, None)

            e_range = (0.0, 1.0)
            e_label = r'$e$'
            if log_e is True:
                e_range = (0.0, np.floor(np.nanmin(e)))
                e_label = r'$\log (1-e)$'

        #Order - P, a, i, e, w, O, tau, T0, d, M, q, Vrest
        if sumdiff is False:
            if rv_a is True:
                samples_plot = np.dstack((P, a_mas, a_au, i, e, w, o, tau, T0_year, d, M, q, vrest))
                axis_labels = [r'$P$ (yrs)', r'$a$ (mas)', r'$a$ (au)', r'$i$', e_label, r'$\omega$', r'$\Omega$', r'$\tau$', r'$T_0$', r'$d$', r'$M_{\rm Total}$', r'$q$', r'$v_{\rm rest}$']
                axis_range = [(p_min, p_max), (a_min, a_max), (a_au_min, a_au_max), (0.0, 180.0), e_range, (0.0, 360.0), (0.0, 360.0), (0.0, 1.0), (T0_min, T0_max), 0.999, 0.999, 0.999, 0.999]
                chain_ind = (1, 3, 4, 5, 6, 7, 9, 10, 11, 12)
            else:
                samples_plot = np.dstack((P, a_mas, a_au, i, e, w, o, tau, T0_year, d, M))
                axis_labels = [r'$P$ (yrs)', r'$a$ (mas)', r'$a$ (au)', r'$i$', e_label, r'$\omega$', r'$\Omega$', r'$\tau$', r'$T_0$', r'$d$', r'$M_{\rm Total}$']
                axis_range = [(p_min, p_max), (a_min, a_max), (a_au_min, a_au_max), (0.0, 180.0), e_range, (0.0, 360.0), (0.0, 360.0), (0.0, 1.0), (T0_min, T0_max), 0.999, 0.999]
                chain_ind = (1, 3, 4, 5, 6, 7, 9, 10)
        else:
            samples_plot = np.dstack((P, a_mas, a_au, i, e, w, o, plus, minus, tau, T0_year, d, M))
            axis_labels = [r'$P$ (yrs)', r'$a$ (mas)', r'$a$ (au)', r'$i$', e_label, r'$\omega$', r'$\Omega$', r'$\Omega+\omega$', r'$\Omega-\omega$',r'$\tau$', r'$T_0$', r'$d$', r'$M_{\rm Total}$']
            axis_range = [(p_min, p_max), (a_min, a_max), (a_au_min, a_au_max), (0.0, 180.0), e_range, (0.0, 360.0), (0.0, 360.0), (0.0, 360.0), (0.0, 360.0), (0.0, 1.0), (T0_min, T0_max), 0.999, 0.999]            
            chain_ind = (1, 3, 4, 7, 8, 9, 11, 12)

        fig, ax = plt.subplots(len(chain_ind), sharex=True, sharey=False, figsize=(12, 10))
        plt.xlabel('Sample')
          
        for j, count in zip(chain_ind, np.arange(len(chain_ind))):
            if sumdiff is True:
                for k in xrange(0, nwalkers*2, int((nwalkers*2)/100)): ax[count].plot(samples_plot[k, :, j], color = 'k', alpha = 0.05, lw = 0.5)
            else:
                for k in xrange(0, nwalkers, int(nwalkers/100)): ax[count].plot(samples_plot[k,:,j], color='k', alpha = 0.05, lw = 0.5)
            ax[count].set_ylabel(axis_labels[j])

        ax[0].set_yscale('log')

        #Trim some of the ranges
        #Inclination
        ylim = list(ax[1].get_ylim())
        if ylim[0] < 0:
            ylim[0] = 0.0
        if ylim[1] > 180.:
            ylim[1] = 180.
        ax[1].set_ylim(ylim)

        #w/plus
        ylim = list(ax[3].get_ylim())
        if ylim[0] < 0.0:
            ylim[0] = 0.0
        if ylim[1] > 360.0:
            ylim[1] = 360.0
        ax[3].set_ylim(ylim)

        #o/minus
        ylim = list(ax[4].get_ylim())
        if ylim[0] < 0.0:
            ylim[0] = 0.0
        if ylim[1] > 360.0:
            ylim[1] = 360.0
        ax[4].set_ylim(ylim)

        if nburn > 0:
            for j, count in zip(chain_ind, np.arange(len(chain_ind))):
                ylim = list(ax[count].get_ylim())
                ax[count].plot((nburn, nburn), (ylim[0], ylim[1]), '-', color='red', zorder=100, lw=2) 
                ax[count].set_ylim(ylim) 

        plt.savefig(object_name+'_chains_'+str(itemp)+'.png', bbox_inches='tight', dpi = 200)
        plt.close()

        #Now plot autocorrelation:
        if itemp == 0:
            fig, ax = plt.subplots(len(chain_ind), sharex = True, sharey=False, figsize=(12, 10))
            for j, count in zip(chain_ind, np.arange(len(chain_ind))):
                if sumdiff is True:
                    for k in xrange(0, nwalkers*2, int((nwalkers*2)/100)): ax[count].plot(emcee.autocorr.function(samples_plot[k, nburn:, j]), color = 'k', alpha = 0.05, lw = 0.5, zorder = 10)
                else:
                    for k in xrange(0, nwalkers, int(nwalkers/100)): ax[count].plot(emcee.autocorr.function(samples_plot[k, nburn:, j]), color='k', alpha = 0.05, lw = 0.5, zorder = 10)
                
                ax[count].plot((0.0, len(samples_plot[k, nburn:, j])), (0.0, 0.0), ':', lw = 1, color = 'red')
                ax[count].set_ylabel(axis_labels[j])
                ax[count].set_ylim(-0.1, 0.1)
                ax[count].set_xlim(0.0, len(samples_plot[k, nburn:, j]))


            plt.savefig(object_name+'_autocorr_'+str(itemp)+'.png', bbox_inches = 'tight', dpi = 200)
            plt.close()

        fig = corner.corner(samples_plot[:,nburn::,:].reshape((-1, ndim)), bins = 50, labels=axis_labels, show_titles=True, plot_datapoints=False,range = axis_range, hist_kwargs={'linewidth': 1.0}, **{'contour_kwargs': {'linewidths': 1.0}})
        fig.savefig(object_name+'_triangle_'+str(itemp)+'.png', dpi=200)
        fig.savefig(object_name+'_triangle_'+str(itemp)+'.pdf')


        if log_e is True:
            e = np.copy(old_e)

        if itemp == 0:
            #At lowest temperature plot 100 best orbits and RVs

            plt.figure(figsize=(15, 7))
            ax1 = plt.subplot2grid((5, 2), (0,0), rowspan = 5)
            ax2 = plt.subplot2grid((5, 2), (0, 1))
            ax3 = plt.subplot2grid((5, 2), (1, 1))
            ax4 = plt.subplot2grid((5, 2), (2, 1),)
            ax5 = plt.subplot2grid((5, 2), (3, 1))
            ax6 = plt.subplot2grid((5, 2), (4, 1))

            for this_ax in (ax1, ax2, ax3, ax4, ax5, ax6):
                this_ax.tick_params(axis='x', labelsize=14)
                this_ax.tick_params(axis='y', labelsize=14)

            ax1.plot([0.0], [0.0], '*', markersize=20.0, color='yellow')

            #Re-shape arrays
            a = a[:,nburn::].flatten()
            e = e[:,nburn::].flatten()
            i = i[:,nburn::].flatten()
            w = w[:,nburn::].flatten()
            o = o[:,nburn::].flatten()
            tau = tau[:,nburn::].flatten()
            plx = plx[:,nburn::].flatten()
            M = M[:,nburn::].flatten()
            d = d[:,nburn::].flatten()
            P = P[:,nburn::].flatten()
            T0 = T0[:,nburn::].flatten()
            T0_year = T0_year[:,nburn::].flatten()

            if rv_a is True:
                q = q[:,nburn::].flatten()
                vrest = vrest[:,nburn::].flatten()

            #Sample lowest temperature walker 
            ind = np.random.randint(len(P), size = 100)

            if date_range is False:
                date_range = [np.floor(np.min(jd2year(epochs, mjd = True))-1.0), np.ceil(np.max(jd2year(epochs, mjd = True))+3.0)]

                if u_limits is not False:
                    u_limits_yrs = jd2year(u_limits[:,0], mjd = True)
                    ax4.plot(u_limits_yrs, u_limits[:,1], 'v', color='k', zorder=10, markersize=5)

                    if np.min(u_limits_yrs) < date_range[0]:
                        date_range = [np.floor(np.min(u_limits_yrs) - 1.0), date_range[1]]
                    if np.max(u_limits_yrs) > date_range[1]:
                        date_range = [date_range[0], np.ceil(np.max(u_limits_yrs)+3.0)]

                
            for j in xrange(0, len(ind)):

                #One complete orbit for x,y plot
                plot_epochs_one = (np.arange(-0.5, 0.5, 0.001)*P[ind[j]]*365.25)+epochs[0]

                #Conditional statement if we are sampling more than one orbit
                if (year2jd(date_range[1], mjd = True) - year2jd(date_range[0], mjd = True)) > (P[ind[j]]*365.25):
                    plot_epochs = np.linspace(year2jd(date_range[0], mjd = True), year2jd(date_range[1], mjd = True), num=2000)
                else:
                    plot_epochs = plot_epochs_one

                plot_epochs_yrs = jd2year(plot_epochs_one, mjd = True)

                if rv_a is True:
                    this_q = q[ind[j]]
                    this_vrest = vrest[ind[j]]
                else:
                    this_q = 1.0
                    this_vrest = 0.0


                if ephems is not False:
                    if j == 0:
                        ephem_rho = np.zeros((len(ephems), len(ind)), dtype=np.float64)
                        ephem_theta = np.zeros((len(ephems), len(ind)), dtype=np.float64)
                        ephem_dx = np.zeros((len(ephems), len(ind)), dtype=np.float64)
                        ephem_dy = np.zeros((len(ephems), len(ind)), dtype=np.float64)
                    
                    #Loop over mjds and print median, 1sigma ranges for rho, theta, dx, dy
                    for k in xrange(0, len(ephems)):
                        x, y, r, t, va, vb = elements_to_xy(P[ind[j]], a[ind[j]], i[ind[j]]*dr, e[ind[j]], w[ind[j]]*dr, o[ind[j]]*dr, T0[ind[j]], ephems[k], rtheta=True, rvs=True, plx = plx[ind[j]], q = this_q, vrest = this_vrest)
                        ephem_rho[k,j] = r
                        ephem_theta[k,j] = t * rd
                        ephem_dx[k,j] = x
                        ephem_dy[k,j] = y

                        if j == 0:
                            print ephem_rho[k,j], ephem_theta[k,j], ephem_dx[k, j], ephem_dy[k,j]

                x, y, r, t, va, vb = elements_to_xy(P[ind[j]], a[ind[j]], i[ind[j]]*dr, e[ind[j]], w[ind[j]]*dr, o[ind[j]]*dr, T0[ind[j]], plot_epochs_one, rtheta=True, rvs=True, plx = plx[ind[j]], q = this_q, vrest = this_vrest)
                ax1.plot(x, y, color='#0082ff', linewidth=1.5, alpha = 0.10)

                plot_epochs_yrs = jd2year(plot_epochs, mjd = True)
                date_ind = np.where((plot_epochs_yrs >= date_range[0]) & (plot_epochs_yrs <= date_range[1]))
                x, y, r, t, va, vb = elements_to_xy(P[ind[j]], a[ind[j]], i[ind[j]]*dr, e[ind[j]], w[ind[j]]*dr, o[ind[j]]*dr, T0[ind[j]], plot_epochs, rtheta=True, rvs=True, plx = plx[ind[j]], q = this_q, vrest = this_vrest)
                ax2.plot(plot_epochs_yrs[date_ind], x[date_ind], color='#0082ff', linewidth=1.5, alpha = 0.10)
                ax3.plot(plot_epochs_yrs[date_ind], y[date_ind], color='#0082ff', linewidth=1.5, alpha = 0.10)
                ax4.plot(plot_epochs_yrs[date_ind], r[date_ind], color='#0082ff', linewidth=1.5, alpha = 0.10)
                ax5.plot(plot_epochs_yrs[date_ind], t[date_ind]*rd, color='#0082ff', linewidth=1.5, alpha = 0.10)
                ax6.plot(plot_epochs_yrs[date_ind], va[date_ind], color='#0082ff', linewidth=1.5, alpha = 0.10)
               
            if ephems is not False:
                for k in xrange(0, len(ephems)):
                    print 'MJD = %10.2f, rho = %6.4f +- %6.4f, theta = %5.3f +- %5.3f, dx = %6.4f +- %6.4f, dy = %6.4f +- %6.4f' % (ephems[k], np.nanmedian(ephem_rho[k,:]), np.nanstd(ephem_rho[k,:]), np.nanmedian(ephem_theta[k,:]), np.nanstd(ephem_theta[k,:]), np.nanmedian(ephem_dx[k,:]), np.nanstd(ephem_dx[k,:]), np.nanmedian(ephem_dy[k,:]), np.nanstd(ephem_dy[k,:]))


            if rho_theta is True:
                dx = rho * np.sin(theta)
                dy = rho * np.cos(theta)
                dx_err = np.zeros(len(dx)) * np.nan
                dy_err = np.zeros(len(dy)) * np.nan

                for k in xrange(0, len(rho)):
                    if np.isfinite(rho[k]):
                        foo_rho = np.random.normal(loc = rho[k], scale = rho_err[k], size=int(1e5))
                        foo_theta = np.random.normal(loc = theta[k], scale = theta_err[k], size=int(1e5))
                        dx_err[k] = np.std(foo_rho * np.sin(foo_theta))
                        dy_err[k] = np.std(foo_rho * np.cos(foo_theta))

            else:
                rho = np.sqrt((dx*dx)+(dy*dy))
                theta = np.arctan2(dy,-dx)
                #Wrap theta
                theta = (theta + (1.5 * np.pi)) % (2 * np.pi)

                rho_err = np.zeros(len(dx), dtype=np.float64)
                theta_err = np.zeros(len(dx), dtype=np.float64)
                for k in xrange(0, len(dx)):
                    foo_x = np.random.normal(dx[k], dx_err[k], size=int(1e5))
                    foo_y = np.random.normal(dy[k], dy_err[k], size=int(1e5))
                    rho_err[k] = np.std(np.sqrt((foo_x**2.0) + (foo_y**2.0)))
                    theta_err[k] = np.std((np.arctan2(foo_y,-foo_x) + (1.5 * np.pi)) % (2 * np.pi))

            ax1.errorbar(dx, dy, xerr = dx_err, yerr = dy_err, fmt='o', color='k', zorder=10, markersize=3, capsize=0)
            ax2.errorbar(jd2year(epochs, mjd = True), dx, yerr = dx_err, fmt='o', color='k', zorder=10, markersize=3, capsize=0)
            ax3.errorbar(jd2year(epochs, mjd = True), dy, yerr = dy_err, fmt='o', color='k', zorder=10, markersize=3, capsize=0)
            ax4.errorbar(jd2year(epochs, mjd = True), rho, yerr = rho_err, fmt='o', color='k', zorder=10, markersize=3, capsize=0)
            ax5.errorbar(jd2year(epochs, mjd = True), theta*rd, yerr = theta_err*rd, fmt='o', color='k', zorder=10, markersize=3, capsize=0)

            if rv_a is True:
                ax6.errorbar(jd2year(epochs, mjd = True), va_obs, yerr = va_obs_err, fmt = 'o', color='k', zorder=10, markersize=3, capsize=0)
            
            #Now select ranges for axes sensibly
            if x_range is False:
                ax1.set_aspect('equal','datalim')
                ax1.set_xlim(ax1.get_xlim()[::-1])
            else:
                ax1.set_xlim(x_range)
                ax1.set_ylim(y_range)

            if r_range is not False:
                ax4.set_ylim(r_range)
            if t_range is not False:
                ax5.set_ylim(t_range)

            #Trim theta range to min = 0, max = 360
            foo = list(ax5.get_ylim())
            if foo[0] < 0:
                foo[0] = 0.0
            if foo[1] > 360.:
                foo[1] = 360.
            ax5.set_ylim(foo)

            for ax in (ax2, ax3, ax4, ax5, ax6):
                ax.set_xlim(date_range)

            for ax in (ax2, ax3, ax4, ax5):
                ax.xaxis.set_ticklabels([])

            ax6.ticklabel_format(useOffset=False, axis='x')

            ax1.set_ylabel(r'$\Delta \delta$ (arc sec)', fontsize = 14)
            ax1.set_xlabel(r'$\Delta \alpha$ (arc sec)', fontsize = 14)

            ax2.set_ylabel(r'$\Delta x$ (arc sec)', fontsize = 14)
            ax3.set_ylabel(r'$\Delta y$ (arc sec)', fontsize = 14)            
            ax4.set_ylabel(r'$\rho$ (arc sec)', fontsize = 14)
            ax5.set_ylabel(r'$\theta$ (deg)', fontsize = 14)

            ax6.set_xlabel('Epoch (Julian year)', fontsize = 14)
            ax6.set_ylabel(r'$v_{\rm a}$ (km s$^{-1}$)', fontsize = 14)

            plt.savefig(object_name+'_orbit.pdf', bbox_inches='tight')
            plt.close()

    plt.close('all')

    return 0

def read_data(data_file, mjd = True, rho_theta = True, rv_a = False):

    dr = (np.pi / 180.0)
    rd = (180.0 / np.pi)
    
    data = ascii.read(data_file, delimiter=',', fill_values=[('', 'None'), (' ', 'None')])
    if mjd is True:
        epochs = np.array((data['mjd'].data).astype(np.float64))
    else:
        #Assuming decimal year
        epochs_year = np.array(data['dec_year'].data).astype(np.float64)
        n = len(epochs_year)
        epochs = np.zeros(n, dtype=np.float64)
        for i in xrange(0,n):
            epochs[i] = year2jd(epochs_year[i], mjd = True)

    if rho_theta is True:
        a = np.array(data['rho'].data).astype(np.float64)
        a_err = np.array(data['rho_err'].data).astype(np.float64)
        b = np.array((data['theta'].data) * dr).astype(np.float64)
        b_err = np.array((data['theta_err'].data) * dr).astype(np.float64)

    else:
        a = np.array(data['dx'].data).astype(np.float64)
        a_err = np.array(data['dx_err'].data).astype(np.float64)
        b = np.array(data['dy'].data).astype(np.float64)
        b_err = np.array(data['dy_err'].data).astype(np.float64)

    if rv_a is True:
        va = np.array(data['va'].data).astype(np.float64)
        va_err = np.array(data['va_err'].data).astype(np.float64)

        return epochs, a, a_err, b, b_err, va, va_err
    else:
        return epochs, a, a_err, b, b_err


def elements_to_xy(P,a,i,e,w,o,T0,epochs, rtheta = False, rvs = False, plx = 0.0, q = 1.0, vrest = 0.0, sumdiff = False):

    # P - Period (years)
    # a - Semi-major axis (asec)
    # i - Inclination (rad)
    # e - Eccentricity
    # w - (omega) Argument of periapsis OR (Omega + omega)
    # o - (Omega) Longitude of ascending node OR (Omega - omega)
    # T0 - Epoch of periapsis (MJD)
    # epcohs - array of MJDs

    #plx = parallax in mas
    #vrest = rest velocity in KM/s
    #sumdiff - if set, w = (Omega + omega), o = (Omega - omega)

    Pdays = P * 365.25
    Tperi = (epochs - T0) % Pdays
    Manom = (2.0 * np.pi) * (Tperi/Pdays)

    if hasattr(Manom, '__len__'):
        n = len(Manom)
        Eanom = kepler_vector(Manom, e)
        Tanom = np.arccos((np.cos(Eanom) - e) / (1.0 - (e * np.cos(Eanom))))
        ind = np.where(Eanom > np.pi)
        Tanom[ind] = (2.0 * np.pi) - Tanom[ind]
    else:
        n = 1
        Eanom = kepler_scalar(Manom, e)
        Tanom = np.arccos((np.cos(Eanom) - e) / (1.0 - (e * np.cos(Eanom))))
        if Eanom > np.pi:
            Tanom = (2.0 * np.pi) - Tanom

    #print 'e = %.8f, M1 = %.8f, M2 = %.8f, M3 = %.8f' % (e, Manom[0], Manom[1], Manom[2]) 

    radius = (a * (1.0 - (e**2.0))) / (1.0 + (e * np.cos(Tanom)))
    if sumdiff is True:
        x = radius * ((np.cos(i/2.0)**2.0 * np.sin(w + Tanom)) - (np.sin(i/2.0)**2.0 * np.sin(Tanom - o)))
        y = radius * ((np.cos(i/2.0)**2.0 * np.cos(w + Tanom)) + (np.sin(i/2.0)**2.0 * np.cos(Tanom - o)))
    else:
        x = radius * ( (np.cos(w + Tanom)*np.sin(o)) + (np.sin(w + Tanom)*np.cos(i)*np.cos(o)))
        y = radius * ( (np.cos(w + Tanom)*np.cos(o)) - (np.sin(w + Tanom)*np.cos(i)*np.sin(o)))
        #z = radius * np.sin(Tanom + w) * np.sin(i)
    
    #x = delta RA
    #y = delta DE, opposite of convention in textbook for x = delta DE

    if rtheta:
        r = np.sqrt((x**2.0) + (y**2.0))
        t = np.arctan2(y,-x)
        #Wrap theta
        t = (t + (1.5 * np.pi)) % (2.0 * np.pi)
    else:
        r, t = np.zeros(n), np.zeros(n)

    if rvs:
        #We know total mass
        #We are fitting for q
        #ka/kb = q
        a_au = a * (1000.0/plx)
        mtot_msol = (a_au**3.0)/(P**2.0)
        mb_msol = (mtot_msol * q)/(1.0 + q)
        ma_msol = mtot_msol - mb_msol 

        aa = (mb_msol/mtot_msol) * a_au
        ab = (ma_msol/mtot_msol) * a_au

        #1 au in km/1 year in seconds = 4.740470463533348
        ka = ((2.0*np.pi*aa*np.sin(i))/(P * np.sqrt(1.0-(e**2.0)))) * 4.740470463533348
        kb = ((2.0*np.pi*ab*np.sin(i))/(P * np.sqrt(1.0-(e**2.0)))) * 4.740470463533348

        va = ( ka * ((e * np.cos(w)) + np.cos(Tanom + w))) + vrest
        vb = (-kb * ((e * np.cos(w)) + np.cos(Tanom + w))) + vrest
    else:
        va, vb = np.zeros(n), np.zeros(n)

    return x, y, r, t, va, vb

def kepler_vector(Manom, e):
    #Solves Kepler's equation on a vector of mean anomalies
    #e < 0.95 use Newton
    #e >= 0.95 use Mikkola
    if e == 0.0:
        return np.copy(Manom)

    if e < 0.95:
        Eanom = np.copy(Manom)
        Eanom -= (Eanom - (e * np.sin(Eanom)) - Manom) / (1.0 - (e * np.cos(Eanom)))
        Eanom -= (Eanom - (e * np.sin(Eanom)) - Manom) / (1.0 - (e * np.cos(Eanom)))

        diff = (Eanom - (e * np.sin(Eanom)) - Manom) / (1.0 - (e * np.cos(Eanom)))
        abs_diff = np.fabs(diff)

        niter = 0
        while True:
            ind = np.where(abs_diff > 1e-9)
            if ind[0].size == 0:
                break
            Eanom[ind] -= diff[ind]
            diff[ind] = (Eanom[ind] - (e * np.sin(Eanom[ind])) - Manom[ind]) / (1.0 - (e * np.cos(Eanom[ind])))
            abs_diff[ind] = np.fabs(diff[ind])
            if niter == 100:
                print Manom[ind], Eanom[ind], e, '> 100 iter.'
                Eanom = kepler_mikkola_vector(Manom, e) # Send it to the analytical version, this has not happened yet...
                break
            niter += 1
    else:
        Eanom = kepler_mikkola_vector(Manom, e)

    return Eanom

def kepler_scalar(Manom, e):
    #Solve Kepler's equation for a scalar

    if e == 0.0:
        Eanom = Manom
    else:
        if e < 0.95:
            Eanom = Manom
            diff = (Eanom - (e * np.sin(Eanom)) - Manom) / (1.0 - (e * np.cos(Eanom)))
            niter = 0
            while ((np.fabs(diff) >= 1e-9) and (niter <= 1e2)):
                Eanom -= diff
                diff = (Eanom - (e * np.sin(Eanom)) - Manom) / (1.0 - (e * np.cos(Eanom)))
                niter += 1
            if niter >= 1e2:
                print Manom, Eanom, e, '> 1e2 iter.'
                Eanom = kepler_mikkola_scalar(Manom, e) # Send it to the analytical version, this has not happened yet...
        else:
            Eanom = kepler_mikkola_scalar(Manom, e)

    return Eanom

def kepler_mikkola_vector(Manom, e):

    ind_change = np.where(Manom > np.pi)
    Manom[ind_change] = (2.0 * np.pi) - Manom[ind_change]
    Eanom = kepler_mikkola(Manom, e)
    Eanom[ind_change] = (2.0 * np.pi) - Eanom[ind_change]

    return Eanom

def kepler_mikkola_scalar(Manom, e):

    if Manom > np.pi:
        Manom = (2.0 * np.pi) - Manom
        Eanom = kepler_mikkola(Manom, e)
        Eanom = (2.0 * np.pi) - Eanom
    else:
        Eanom = kepler_mikkola(Manom, e)
    return Eanom

def kepler_mikkola(Manom, e):

    #Adapted from IDL routine keplereq.pro
    #http://www.lpl.arizona.edu/~bjackson/idl_code/keplereq.pro

    alpha = (1.0 - e) / ((4.0 * e) + 0.5)
    beta = (0.5 * Manom) / ((4.0 * e) + 0.5)

    aux = np.sqrt(beta**2.0 + alpha**3.0)
    z = (beta + aux)**(1.0/3.0)

    s0 = z - (alpha/z)
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + e)
    e0 = Manom + (e * (3.0*s1 - 4.0*(s1**3.0)))

    se0=np.sin(e0)
    ce0=np.cos(e0)

    f  = e0-e*se0-Manom
    f1 = 1.0-e*ce0
    f2 = e*se0
    f3 = e*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.16666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.16666666666667*f3*u3*u3+0.041666666666667*f4*(u3**3.0))

    return (e0 + u4)
    
def year2jd(year, mjd=False):
    #Leads to innacuracies between -0.5 and 0.5 days. Will fix
    j2000 = 2451545.0
    cj = 36525.0

    jd = (j2000 + ((np.double(year) - 2000.0) * (cj / 100.0)))
    
    if mjd == True:
        jd -= 2400000.5

    return jd

def jd2year(jd, mjd=False):

    j2000 = 2451545.0
    cj = 36525.0

    if mjd is False:
        year = ((jd - j2000) / (cj / 100.0)) + 2000.0
    else:
        year = (((jd + 2400000.5) - j2000) / (cj / 100.0)) + 2000.0

    return year

