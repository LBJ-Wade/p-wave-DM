print "initializing..."
#Math packages, NumPy
import math
import numpy as np
from scipy.special import gammainc, erf, gamma
from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
import scipy
from scipy.optimize import curve_fit

import pickle

#Plotting packages ie Matplotlib stuff
import pylab as plt
from matplotlib import cm, colors
from matplotlib.pyplot import rc
from matplotlib import rcParams
import pyfits
#from astropy import wcs

#Fermi Science Tools
from gt_apps import srcMaps
from gt_apps import evtbin
from gt_apps import gtexpcube2
from BinnedAnalysis import *
#from UnbinnedAnalysis import *
from SummedLikelihood import *
import pyLikelihood as pyLike
#import residmaps

from astropy.io import fits
from gt_apps import evtbin, filter
print "Done!"

def setup_plot_env():
    #Set up figure
    #Plotting parameters
    fig_width = 8   # width in inches
    fig_height = 8  # height in inches
    fig_size =  [fig_width, fig_height]
    rcParams['font.family'] = 'serif'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelsize'] = 20
    rcParams['font.size'] = 20
    rcParams['axes.titlesize'] =16
    rcParams['legend.fontsize'] = 16
    rcParams['xtick.labelsize'] =20
    rcParams['ytick.labelsize'] =20
    rcParams['figure.figsize'] = fig_size
    rcParams['xtick.major.size'] = 8
    rcParams['ytick.major.size'] = 8
    rcParams['xtick.minor.size'] = 4
    rcParams['ytick.minor.size'] = 4
    rcParams['xtick.major.pad'] = 8
    rcParams['ytick.major.pad'] = 8
    
    rcParams['figure.subplot.left'] = 0.16
    rcParams['figure.subplot.right'] = 0.92
    rcParams['figure.subplot.top'] = 0.90
    rcParams['figure.subplot.bottom'] = 0.12
    rcParams['text.usetex'] = True
    rc('text.latex', preamble=r'\usepackage{amsmath}')
setup_plot_env()


#Strictly finds the Poisson upper limit (problematic for large counts)
def upper_limit(N,conf,b):
    #Calculates upper limit on signal s, given background b, number of events N, and confidence interval conf (95% = 0.05)
    #Decides whether the value is really big, in which case do everything with ints. Otherwise, use floats
    #First, calculate denominator
    denom=0.
    for m in range(0,N+1):
            denom+=b**m/math.factorial(m)
    s = 0.
    numer=denom
    while math.exp(-1.0*s)*numer/denom>conf:
        #Calculate numerator
        numer=0.0
        for m in range(0,N+1):
            numer+=(s+b)**m/math.factorial(m)
        s+= 0.01
    print "Upper limit is " + str(s)
    return s

def factorial2(x):
	result = 1.
	while x>0:
		result *= x
		x -= 1.
	return result
def gamma(x):
	if x%1 == 0:
		return factorial(x-1)
	if x%1 == 0.5:
		return np.sqrt(np.pi)*factorial(2*(x-0.5))/(4**(x-0.5)*factorial((x-0.5)))
def chi_square_pdf(k,x):
    return 1.0/(2**(k/2)*gamma(k/2))*x**(k/2-1)*np.exp(-0.5*x)
def chi_square_cdf(k,x):
    return gammainc(k/2,x/2)
    
def chi_square_quantile(k,f):
    #Essentially do a numerical integral, until the value is greater than f
    integral_fraction = 0.0
    x = 0.0
    dx = 0.01
    while chi_square_cdf(k,x)<f:
        x += dx
    return x
def upper_limit_pdg(N,alpha,b):
    dof = 2*(N+1)
    p = 1-alpha*(1-(chi_square_cdf(dof, 2*b)))
    sup = 0.5*chi_square_quantile(dof, p)-b
    return sup
def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))

def chisquared(data,theory):
    thesum = 0.0
    for i in range(len(data)):
        if theory[i]>0.0:
            thesum += (theory[i]-data[i])**2/(float(theory[i]))
    return thesum
def loglikelihood(counts,model_counts,box):
    f0 = 0.0
    f1 = 0.0
    for i in range(len(counts)):
        #m = number of counts
        m = counts[i]
        #null hypothesis counts:
        mu0 = model_counts[i]
        #background+signal counts
        mu1 = model_counts[i]+box[i]
        #Do we need stirling's approximation?
        if m>20:
            f0 += m-mu0+m*np.log(mu0/m)-0.5*np.log(m)-np.log(np.sqrt(2*np.pi))
            f1 += m-mu1+m*np.log(mu1/m)-0.5*np.log(m)-np.log(np.sqrt(2*np.pi))
        else:
            f0 += np.log((mu0**m)*np.exp(-1.0*mu0)/factorial(m))
            f1 += np.log((mu1**m)*np.exp(-1.0*mu1)/factorial(m))
    return 2*(f0-f1)
def sigma_given_p(p):
    x = np.linspace(-200, 200, 50000)
    g = 1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.)
    c = np.cumsum(g)/sum(g)
    value = x[np.argmin(np.abs(c-(1.0-p)))]
    return value
def pvalue_given_chi2(x, k):
    y = np.arange(0., 1000.0, 0.1)
    g = (y**(k/2.-1.0)*np.exp(-0.5*y))/(2.**(k/2.0)*gamma(k/2.))
    initial_pos = np.argmin(np.abs(y-x))
    return get_integral(y[initial_pos:], g[initial_pos:])
#Stupid function bc apparently numpy can't do this natively??
def multiply_multidimensional_array(vec,cube):
    result = np.zeros((cube.shape))
    for i in range(len(vec)):
        result[i,:,:] = vec[i]*cube[i,:,:]
    return result
#Energy resolution (E Disp class 1- pretty close to the total)
def e_res(E):
    energy = np.array([31.718504,54.31288,95.86265,171.82704,307.93054,535.1251,944.0604,1716.8848,3074.5315,5339.1763,9559.056,17111.936,29704.664,53986.227,93706.14,167702.28,300152.7,520977.8,931974.8,1690808.0,2974747.2])
    res = np.array([0.2942218,0.25078142,0.21741302,0.1813951,0.15067752,0.12313887,0.10302135,0.090325624,0.08080946,0.07341215,0.07025641,0.070810914,0.07454435,0.080399856,0.08678346,0.094758466,0.10061333,0.10752697,0.25969607,0.18390165,0.24169934])
    closest = np.argmin(np.abs(E-energy))
    if E-energy[closest]>0.:
        frac = (E-energy[closest])/(energy[closest+1]-energy[closest])
        return res[closest]+frac*(res[closest+1]-res[closest])
    else:
        frac = (E-energy[closest-1])/(energy[closest]-energy[closest-1])
        return res[closest-1]+frac*(res[closest]-res[closest-1])
def psf(E):
    energy = np.array([9.91152,17.36871,31.150045,54.59528,96.42895,171.62605,303.14316,539.58026,967.85913,1709.5619,3066.256,5374.1895,9712.058,17151.041,29366.348,52649.074,92947.98,167911.25,298723.0,527422.3,952855.0,1682382.6,2993103.8])
    psf = np.array([22.122343,17.216175,11.960119,8.108732,5.279108,3.5216076,2.2375877,1.3988715,0.8535155,0.53358656,0.347393,0.23173566,0.17039458,0.12837319,0.112826064,0.10581638,0.10334797,0.10426899,0.10101496,0.09097172,0.08671612,0.07683781,0.073241934])
    closest = np.argmin(np.abs(E-energy))
    if E-energy[closest]>0.:
        frac = (E-energy[closest])/(energy[closest+1]-energy[closest])
        return psf[closest]+frac*(psf[closest+1]-psf[closest])
    else:
        frac = (E-energy[closest-1])/(energy[closest]-energy[closest-1])
        return psf[closest-1]+frac*(psf[closest]-psf[closest-1])
#Just a gaussian function, representing the energy dispersion of the detector
def blur(x,offset,sigma):
    return np.exp(-1.0*(x-offset)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def make_random(x,g):
    cdf = np.cumsum(g)/np.sum(g)
    return x[np.argmin(np.abs(cdf-np.random.rand(1)[0]))]
def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))

    
def skew_gaussian(x, norm, x_bar, sigma, alpha):
    return 2.0*norm*np.exp(-1.0*(x-x_bar)**2/(2.0*sigma**2))*(1+erf(alpha*(x-x_bar)/sigma/np.sqrt(2)))
    
num_ebins = 51
energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
ebin_widths = np.diff(energies)



def edit_box_xml(energy, box_flux):
    #Choose between a wide and narrow box
    box_beginning = energy-100# -> 100 MeV wide box
    
    #Edit box_spectrum.dat
    file = open('box_spectrum.dat','w')
    #What's the normalization constant here?
    #Total flux is E_edge*function value
    #Function_value = total_flux/E_edge
    x_fine_grid = np.linspace(0.0, 800000, 10000)
    n_leading_zeros = int(np.argmin(np.abs(x_fine_grid-box_beginning)))
    n_trailing_zeros = 10000-int(np.argmin(np.abs(x_fine_grid-energy)))
    n_box_width = int(10000-n_leading_zeros-n_trailing_zeros)
    pure_box = np.concatenate([np.zeros((n_leading_zeros)),np.concatenate([1.0+np.zeros((n_box_width)),np.zeros((n_trailing_zeros))])])

    #Sigma here is the absolute energy resolution as a function of energy
    sigma = e_res(energy)*energy*10000./800000.0

    dispersion = blur(np.linspace(0,6*sigma,6*sigma),3*sigma,sigma)
    convolved_pure_box = np.convolve(pure_box, dispersion,'same')
    for i in range(len(x_fine_grid)):
        file.write(str(x_fine_grid[i])+" " + str(max(convolved_pure_box[i], 10.**-35))+"\n")
    file.close()

    file = open('xmlmodel_fixed.xml','r')
    non_box_string = []
    for line in file:
        non_box_string.append(line)
    file.close()

    box_string = []
    box_string.append(' <source name="Box_Component" type="PointSource">\n')
    box_string.append('   <spectrum file="box_spectrum.dat" type="FileFunction">\n')
    box_string.append('     <parameter free="0" max="1e5" min="1e-35" name="Normalization" scale="1" value="'+str(box_flux/(energy-box_beginning))+'"/>\n')
    box_string.append('   </spectrum>\n')
    box_string.append('  <spatialModel type="SkyDirFunction">\n')
    box_string.append('     <parameter free="0" max="360" min="-360" name="RA" scale="1" value="266.417" />\n')
    box_string.append('     <parameter free="0" max="90" min="-90" name="DEC" scale="1" value="-29.0079" />\n')
    box_string.append('   </spatialModel>\n')
    box_string.append(' </source>\n')

    file = open('xmlmodel_fixed_box.xml','w')

    for entry in non_box_string[:-1]:
        file.write(entry)
    for entry in box_string:
        file.write(entry)
    file.write('</source_library>\n')
    file.close()

#Likelihood from gtlikelihood module
def loglikelihood3(energy, box_flux, obs):

    #Next, edit the XML file to have the right flux and spectrum
    edit_box_xml(energy, box_flux)

    #Do the fitting
    like1 = BinnedAnalysis(obs, 'xmlmodel_fixed_box.xml', optimizer='NEWMINUIT')
    like1.tol=1e-8
    like1obj = pyLike.Minuit(like1.logLike)
    q = like1.fit(verbosity=0,optObject=like1obj)
    return q

def get_spectrum(likelihood_obj):
    spectrum = np.zeros((num_ebins-1))
    for source in likelihood_obj.sourceNames():
        spectrum += likelihood_obj._srcCnts(source)
    return spectrum



def likelihood_upper_limit3():
    #Array to hold results of flux upper limit calculation
    num_ebins = 51 #1 more than the number of bins due to the fencepost problem
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
    ebin_widths = np.diff(energies)
    
    sourcemap = '6gev_srcmap_03.fits'#'box_srcmap_artificial_box.fits'#'6gev_srcmap_complete.fits'
    box_flux = np.zeros((num_ebins-1))
    best_box = np.zeros((num_ebins-1))

    gll_counts = np.zeros((num_ebins-1))

    #reconstructed_spectra = np.zeros((num_ebins-1, num_ebins-1))
    #Loop through upper edge of box
    for index in range(18,48):
        print "Calculating upper limit in bin " + str(index) + " at energy " + str(energies[index])
        #print "bin " + str(np.argmin(np.abs(energies-energy)))
        #window_low, window_high = window(energy, energies)
        
        window_low = index-6
        window_high = index+2
        print "window low = " + str(window_low)
        print "window high = " + str(window_high)

        #Generate two observations (one above the window and one below)
        #Make two exposure maps
        if index>6:
            exposure_complete = pyfits.open('6gev_exposure.fits')
            exposure_complete[0].data = exposure_complete[0].data[:window_low+1]
            a = exposure_complete[0]
            exposure_complete[1].data = exposure_complete[1].data[:window_low+1]
            b = exposure_complete[1]
            hdulist = pyfits.HDUList([a, b, exposure_complete[2]])
            os.system('rm exposure_low.fits')
            hdulist.writeto('exposure_low.fits')
            exposure_complete.close()
        if index<48:
            exposure_complete = pyfits.open('6gev_exposure.fits')
            exposure_complete[0].data = exposure_complete[0].data[window_high:]
            a = exposure_complete[0]
            exposure_complete[1].data = exposure_complete[1].data[window_high:]
            b = exposure_complete[1]
            hdulist = pyfits.HDUList([a, b, exposure_complete[2]])
            os.system('rm exposure_high.fits')
            hdulist.writeto('exposure_high.fits')
            exposure_complete.close()
        
        
        #Make two sourcemaps
        if index>6:
            srcmap_complete = pyfits.open(sourcemap)
            srcmap_complete[0].data = srcmap_complete[0].data[:window_low]
            a = srcmap_complete[0]
            srcmap_complete[2].data = srcmap_complete[2].data[:window_low]
            b = srcmap_complete[2]
            srcmap_complete[3].data = srcmap_complete[3].data[:window_low+1]
            c = srcmap_complete[3]
            srcmap_complete[4].data = srcmap_complete[4].data[:window_low+1]
            d = srcmap_complete[4]
            srcmap_complete[5].data = srcmap_complete[5].data[:window_low+1]
            e = srcmap_complete[5]
            srcmap_complete[6].data = srcmap_complete[6].data[:window_low+1]
            f = srcmap_complete[6]
            srcmap_complete[7].data = srcmap_complete[7].data[:window_low+1]
            g = srcmap_complete[7]
            srcmap_complete[8].data = srcmap_complete[8].data[:window_low+1]
            h = srcmap_complete[8]

            os.system('rm srcmap_low.fits')
            b.header['DSVAL4'] = str()+':'+str()
            hdulist = pyfits.HDUList([a, srcmap_complete[1], b, c, d, e, f, g, h])
            hdulist.writeto('srcmap_low.fits')
            srcmap_complete.close()
        
        if index<48:
            srcmap_complete = pyfits.open(sourcemap)
            srcmap_complete[0].data = srcmap_complete[0].data[window_high:]
            a = srcmap_complete[0]
            srcmap_complete[2].data = srcmap_complete[2].data[window_high:]
            r = 0
            for entry in srcmap_complete[2].data:
                entry[0] = int(r)
                r += 1
            #srcmap_complete[2].data[:,0] = np.arange(0, len(srcmap_complete[2].data[:,0]))
            b = srcmap_complete[2]
            srcmap_complete[3].data = srcmap_complete[3].data[window_high:]
            c = srcmap_complete[3]
            srcmap_complete[4].data = srcmap_complete[4].data[window_high:]
            d = srcmap_complete[4]
            srcmap_complete[5].data = srcmap_complete[5].data[window_high:]
            e = srcmap_complete[5]
            srcmap_complete[6].data = srcmap_complete[6].data[window_high:]
            f = srcmap_complete[6]
            srcmap_complete[7].data = srcmap_complete[7].data[window_high:]
            g = srcmap_complete[7]
            srcmap_complete[8].data = srcmap_complete[8].data[window_high:]
            h = srcmap_complete[8]

            os.system('rm srcmap_high.fits')
            hdulist = pyfits.HDUList([a, srcmap_complete[1], b, c, d, e, f, g, h])
            hdulist.writeto('srcmap_high.fits')
            srcmap_complete.close()

        summedLike = SummedLikelihood()

        if index>6:
            obs_low = BinnedObs(srcMaps='/Users/christian/physics/p-wave/6gev/srcmap_low.fits', expCube='/Users/christian/physics/p-wave/6gev/6gev_ltcube.fits', binnedExpMap='/Users/christian/physics/p-wave/6gev/exposure_low.fits', irfs='CALDB')
            like_low = BinnedAnalysis(obs_low, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
            summedLike.addComponent(like_low)


        if index<48:
            obs_high = BinnedObs(srcMaps='/Users/christian/physics/p-wave/6gev/srcmap_high.fits', expCube='/Users/christian/physics/p-wave/6gev/6gev_ltcube.fits', binnedExpMap='/Users/christian/physics/p-wave/6gev/exposure_high.fits', irfs='CALDB')
            like_high = BinnedAnalysis(obs_high, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
            summedLike.addComponent(like_high)

        summedLike.ftol = 1e-8
        summedLike.fit(verbosity=0)
        summedLike.writeXml('xmlmodel_free.xml')
        for k in range(len(summedLike.params())):
            summedLike.freeze(k)
        summedLike.writeXml('xmlmodel_fixed.xml')
        
        obs_complete = BinnedObs(srcMaps='/Users/christian/physics/p-wave/6gev/'+sourcemap, expCube='/Users/christian/physics/p-wave/6gev/6gev_ltcube.fits', binnedExpMap='/Users/christian/physics/p-wave/6gev/6gev_exposure.fits', irfs='CALDB')
        like = BinnedAnalysis(obs_complete, 'xmlmodel_fixed.xml', optimizer='NEWMINUIT')
        like.tol=1e-8
        like_obj = pyLike.Minuit(like.logLike)
        like.fit(verbosity=3,optObject=like_obj)
        complete_spectrum = like.nobs#
        window_spectrum = get_spectrum(like)
        
        #Flucuate the window data
        f = pyfits.open(sourcemap)
        poisson_data = np.zeros((len(range(max(window_low,0), min(window_high, 49))), 50, 50))
        q = 0
        for bin in range(max(window_low,0), min(window_high,49)):
            for source in like.sourceNames():
                for j in range(3,9):
                    if source == f[j].header['EXTNAME']:
                        the_index = j
                    model_counts = np.zeros((1,len(f[the_index].data[bin].ravel())))[0]
                    num_photons = int(np.round(np.random.poisson(like._srcCnts(source)[bin])))#
                for photon in range(int(num_photons)):
                    phot_loc = int(make_random(np.arange(0,len(model_counts), 1), f[the_index].data[bin].ravel()))
                    model_counts[phot_loc] += 1
                model_counts = model_counts.reshape(50, 50)
                poisson_data[q] += model_counts
            q += 1
        f[0].data[max(window_low,0):min(window_high, 49)] = poisson_data
        os.system('rm box_srcmap_poisson.fits')
        f.writeto('box_srcmap_poisson.fits')
        f.close()

        obs_poisson = BinnedObs(srcMaps='box_srcmap_poisson.fits', expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='CALDB')
        like = BinnedAnalysis(obs_poisson, 'xmlmodel_fixed.xml', optimizer='NEWMINUIT')
        like.tol=1e-8
        like_obj = pyLike.Minuit(like.logLike)
        like.fit(verbosity=0,optObject=like_obj)
        poisson_spectrum = like.nobs

        obs_calculation = obs_complete #obs_poisson or obs_complete
        """
        #Plot likelihood profile
        mylikelihood = np.zeros((100))
        b = 0
        for boxflux in 10**np.linspace(-11, -7.5, 100):
            mylikelihood[b] = loglikelihood3(50000, boxflux, obs_calculation)
            b += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xscale('log')
        plt.plot(10**np.linspace(-11, -7.5, 100), -1.0*mylikelihood, linewidth=2, color='blue')
        plt.axvline(2.*10**-9, linestyle='--', color='black')
        plt.show()
        raw_input('wait for key')
        """
        null_likelihood = loglikelihood3(energies[index], 10.**-25, obs_calculation)
        #Find best fit box:
        old_likelihood=null_likelihood
        box_flux[index] = 10.**-13
        print loglikelihood3(energies[index], box_flux[index], obs_calculation)
        while loglikelihood3(energies[index], box_flux[index], obs_calculation)<old_likelihood:
            old_likelihood=loglikelihood3(energies[index], box_flux[index], obs_calculation)
            box_flux[index] *= 1.3
        print "null likelihood = " + str(null_likelihood)
        print "new likelihood = " + str(old_likelihood)
        best_box[index] = box_flux[index]/1.3
        print "best box = " + str(best_box[index]) + ", with significance " + str(sigma_given_p(pvalue_given_chi2(-2.0*(old_likelihood-null_likelihood),3))) + " sigma"
        
        #increase box flux until likelihood > 2sigma over null likelihood
        box_flux[index] = 10.**-13
        while sigma_given_p(pvalue_given_chi2(2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood),3))<2.0:
            print "flux = " + str(box_flux[index]) + " likelihood= " + str(loglikelihood3(energies[index], box_flux[index], obs_calculation))
            box_flux[index]*=3.0
        box_flux[index]*=1.0/3.0
        while sigma_given_p(pvalue_given_chi2(2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood),3))<2.0:
            print "flux = " + str(box_flux[index]) + " likelihood= " + str(loglikelihood3(energies[index], box_flux[index], obs_calculation))
            box_flux[index]*=1.1
        box_flux[index]*=1.0/1.1
        while sigma_given_p(pvalue_given_chi2(2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood),3))<2.0:
            print "flux = " + str(box_flux[index]) + " likelihood= " + str(loglikelihood3(energies[index], box_flux[index], obs_calculation))
            box_flux[index]*=1.03

        print box_flux[index]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(energies[:-1], window_spectrum, linewidth=2., color='blue', label='Reconstructed spectrum')
        plt.errorbar(energies[:-1], poisson_spectrum, xerr=0, yerr=np.sqrt(poisson_spectrum), fmt='o', color='red', label='Poisson fluctuations')
        plt.errorbar(energies[:-1], complete_spectrum, xerr=0, yerr=np.sqrt(complete_spectrum), fmt='o', color='black',label='Actual data')
        plt.axvspan(energies[window_low]-0.5*ebin_widths[window_low], energies[window_high]-0.5*ebin_widths[window_high], alpha=0.3, color='black', label='Window')
        plt.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([8000.0, 800000])
        plt.title('Bin ' + str(index))
        plt.show()
        raw_input('wait for key')
        
        
    return box_flux, best_box#box_flux#, reconstructed_spectra




def consolidate_brazil_lines():
    file = open('brazil_wide_box.pk1','rb')
    g = pickle.load(file)
    file.close()
    brazil_dict = np.zeros((6100,num_ebins-1))
    i = 0
    for entry in g:
        brazil_dict[i,:] = entry
        i += 1
    return brazil_dict


def make_ul_plot(ul, brazil_dict, plot_type):
    mc_limits = brazil_dict[np.nonzero(brazil_dict[:,10])]
    trials = len(mc_limits)
    print "trials = " + str(trials)
    lower_95 = np.zeros((num_ebins-1))
    lower_68 = np.zeros((num_ebins-1))
    upper_95 = np.zeros((num_ebins-1))
    upper_68 = np.zeros((num_ebins-1))
    median = np.zeros((num_ebins-1))
    for i in range(num_ebins-1):
        lims = mc_limits[:,i]
        lims.sort()
        lower_95[i] = lims[int(0.025*trials)]
        upper_95[i] = lims[int(0.975*trials)]
        lower_68[i] = lims[int(0.15865*trials)]
        upper_68[i] = lims[int(0.84135*trials)]
        median[i] = lims[int(0.5*trials)]
    print "Median = " + str(median)
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111)
    print ul
    if plot_type=='UL':
        #Plotting uppper limit
        #ax.plot(energies[:-1],median,color='black',linewidth=1,linestyle='--', label='Median MC')
        ax.fill_between(energies[:-1], lower_95, upper_95, color='yellow', label='95\% Containment')
        ax.fill_between(energies[:-1], lower_68, upper_68, color='#63ff00',label='68\% Containment')
        ax.plot(energies[:-1],ul, marker='.', markersize=13.0,color='black',linewidth=2, label='Data')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel('Flux Upper Limit [ph s$^{-1}$ cm$^{-2}$]')
        plt.xlabel('Energy [MeV]')
        plt.legend()
        
        ax.set_xlim([10000, 600000])
        ax.set_ylim([10**-11, 10**-9])
        plt.savefig('plots/upper_limit_artificial_box.pdf',bbox_inches='tight')
        plt.show()
        
    if plot_type=='SIG':
        #Plotting significance curve
    
        ax.fill_between(energies[:-1], -2.0, 2.0, color='yellow', label='95\% Containment')
        ax.fill_between(energies[:-1], -1.0, 1.0, color='#63ff00', label='68\% Containment')
        ax.axhline(0.0, color='black', linestyle='--', linewidth=0.5)

        plt.ylabel('Significance [$\sigma$]')
        ax.set_xscale('log')
        ax.set_xlim([10000, 600000])
        ax.set_ylim([-4.0, 4.0])
        plt.xlabel('Energy [MeV]')
        plt.legend()
        significances = np.zeros((len(ul)))
        for i in range(6,48):
            print i
            bins = np.linspace(np.min(brazil_dict[:,i][np.nonzero(brazil_dict[:,i])]), np.max(brazil_dict[:,i][np.nonzero(brazil_dict[:,i])]), 25)
            lim_hist = np.histogram(brazil_dict[:,i],bins) 
            #popt, pcov = curve_fit(skew_gaussian, bins[:-1], lim_hist[0], p0=popt)
            popt, pcov = curve_fit(skew_gaussian, bins[:-1], lim_hist[0], p0=[50.0, np.mean(brazil_dict[:,i][np.nonzero(brazil_dict[:,i])]), np.std(brazil_dict[:,i][np.nonzero(brazil_dict[:,i])]), 4.0])
        
            x_range = 10**np.linspace(np.log10(np.min(brazil_dict[:,i][np.nonzero(brazil_dict[:,i])])), 0.0, 5000)
            small_range = 10**np.linspace(np.log10(ul[i]), 0.0, 5000)
            p_value = get_integral(small_range,skew_gaussian(small_range, *popt) )/get_integral(x_range, skew_gaussian(x_range, *popt))
            if p_value>0.5:
                significances[i] = -1.0*sigma_given_p(1.0-p_value)
            else:
                significances[i] = sigma_given_p(p_value)
        print significances

        #significances = np.delete(significances, 41)
        #significances = np.delete(significances, 47)
    
        #energies = np.delete(energies, 41)
        print significances
        print significances.shape
        print energies
        plt.plot(energies[:-1], significances, color='black', marker='.', linewidth=2, markersize=13.0, label='Data')
        plt.savefig('plots/significance_narrow_box.pdf',bbox_inches='tight')
        plt.show()


"""
box_flux = likelihood_upper_limit3()
print box_flux
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(energies[:-1], box_flux, color='black', linewidth=2.0)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([6000.0, 800000])

#ax.set_ylim([10**-11, 10**-8])
plt.xlabel('Energy [MeV]')

plt.show()
raw_input('wait for key')
"""

remake_ul = False

if remake_ul:
    box_flux, best_box = likelihood_upper_limit3()
    raw_input('wait for key')
    
    file = open('upper_limit_narrow_box_mc.pk1', 'wb')
    pickle.dump(box_flux,file)
    file.close()

else:
    file = open('upper_limit_wide_box.pk1', 'rb')
    box_flux = pickle.load(file)
    file.close()

#file = open('brazil.pk1', 'rb')
#brazil_dict = pickle.load(file)
#file.close()

brazil_dict = consolidate_brazil_lines()

make_ul_plot(box_flux, brazil_dict, plot_type='UL')

