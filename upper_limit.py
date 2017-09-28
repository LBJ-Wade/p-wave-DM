print "initializing..."
#Math packages, NumPy
import math
import numpy as np
from scipy.special import gammainc, erf, gamma
from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
import scipy
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import pickle
import psutil
#Plotting packages ie Matplotlib stuff
import pylab as plt
from matplotlib import cm, colors
from matplotlib.pyplot import rc
from matplotlib import rcParams
import matplotlib as mpl
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
from UpperLimits import UpperLimit

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
        
def my_exp(x, norm, k):
    return norm*np.exp(-k*x)
num_ebins = 51
energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
ebin_widths = np.diff(energies)



def edit_box_xml(energy, box_flux, z):
    #Choose between a wide and narrow box
    #z=0: wide box
    #z=1: line
    
    box_width = energy*2.0*np.sqrt(1.0-z)/(1+np.sqrt(1.0-z))
    box_beginning = energy-box_width
    #Edit box_spectrum.dat
    spectrum_file = open('box_spectrum.dat','w')
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
        spectrum_file.write(str(x_fine_grid[i])+" " + str(max(convolved_pure_box[i], 10.**-35))+"\n")
    spectrum_file.close()
    """
    plt.plot(x_fine_grid, convolved_pure_box)
    plt.axvline(energy, linestyle='--',linewidth=0.5, color='black')
    plt.axvspan(box_beginning, energy, alpha=0.5, color='green')
    plt.axvline(box_beginning-sigma, linestyle=':', color='black')
    plt.axvline(energy+sigma, linestyle=':', color='black')
    plt.plot(x_fine_grid, pure_box, color='red')
    print max(blur(x_fine_grid,energy,e_res(energy)*energy))
    plt.plot(x_fine_grid, 10**4*blur(x_fine_grid,energy,e_res(energy)*energy), color='yellow',linestyle='-.')
    plt.show()
    """
    fixed_xml_file = open('xmlmodel_fixed.xml','r')
    non_box_string = []
    for line in fixed_xml_file:
        non_box_string.append(line)
    fixed_xml_file.close()

    box_minimum = 1e-15
    box_maximum = 1e-6
    scale_factor = 1e-15
    
    box_string = []
    box_string.append(' <source name="Box_Component" type="PointSource">\n')
    box_string.append('   <spectrum file="box_spectrum.dat" type="FileFunction">\n')
    box_string.append('     <parameter free="0" max="'+str(box_maximum*scale_factor**-1/box_width)+'" min="'+str(box_minimum*scale_factor**-1/box_width)+'" name="Normalization" scale="'+str(scale_factor)+'" value="'+str(box_flux*scale_factor**-1/box_width)+'"/>\n')
    box_string.append('   </spectrum>\n')
    box_string.append('  <spatialModel type="SkyDirFunction">\n')
    box_string.append('     <parameter free="0" max="360" min="-360" name="RA" scale="1" value="266.417" />\n')
    box_string.append('     <parameter free="0" max="90" min="-90" name="DEC" scale="1" value="-29.0079" />\n')
    box_string.append('   </spatialModel>\n')
    box_string.append(' </source>\n')

    pedestal_present=False
    if pedestal_present:
        pedestal_maximum = 1e-8
        pedestal_minimum = 0.0
    
        box_string.append(' <source name="Pedestal" type="PointSource">\n')
        box_string.append('   <spectrum file="box_spectrum.dat" type="FileFunction">\n')
        box_string.append('     <parameter free="0" max="'+str(pedestal_maximum*scale_factor**-1/box_width)+'" min="'+str(pedestal_minimum*scale_factor**-1/box_width)+'" name="Normalization" scale="'+str(-1*scale_factor)+'" value="'+str(pedestal_minimum*scale_factor**-1/box_width)+'"/>\n')
        box_string.append('   </spectrum>\n')
        box_string.append('  <spatialModel type="SkyDirFunction">\n')
        box_string.append('     <parameter free="0" max="360" min="-360" name="RA" scale="1" value="266.417" />\n')
        box_string.append('     <parameter free="0" max="90" min="-90" name="DEC" scale="1" value="-29.0079" />\n')
        box_string.append('   </spatialModel>\n')
        box_string.append(' </source>\n')

    box_xml_file = open('xmlmodel_fixed_box.xml','w')

    for entry in non_box_string[:-1]:
        box_xml_file.write(entry)
    for entry in box_string:
        box_xml_file.write(entry)
    box_xml_file.write('</source_library>\n')
    box_xml_file.close()

#Likelihood from definition
def loglikelihood2(energy, box_flux, obs, z):
    edit_box_xml(energy, box_flux, z)
    like1 = BinnedAnalysis(obs, 'xmlmodel_fixed_box.xml', optimizer='DRMNFB')
    
    sourcemap = obs.srcMaps
    f = pyfits.open(sourcemap)
    model_data = np.zeros((50, 50, 50))
    like = BinnedAnalysis(obs, 'xmlmodel_fixed_box.xml', optimizer='NEWMINUIT')
    
    for source in like.sourceNames():
        for j in range(3,9):#Change to 10 for pedestal sourcemap
            if source == f[j].header['EXTNAME']:
                the_index = j
        for w in range(50):
            model_data[w, :, :] += like._srcCnts(source)[w]*f[the_index].data[w]/np.sum(np.sum(f[the_index].data[w]))
    actual_data = np.array(obs.countsMap.data()).reshape(50,50,50)
    
    #fig = plt.figure()
    #ax = fig.add_subplot(121)
    #ax.imshow(np.sum(actual_data, axis=0), vmin=0, vmax=60, cmap=cm.inferno, norm=colors.PowerNorm(gamma=0.6))
    #ax2 = fig.add_subplot(122)
    #ax2.imshow(np.sum(model_data, axis=0), vmin=0, vmax=60, cmap=cm.inferno, norm=colors.PowerNorm(gamma=0.6))
    #plt.show()
    likelihood = np.sum(np.sum(np.sum(np.log(model_data**actual_data*np.exp(-1.0*model_data)/factorial(actual_data)))))
    
    return -1.0*likelihood

#Likelihood from gtlikelihood module
def loglikelihood3(energy, box_flux, obs, z, get_like=False, covar=False):

    #Next, edit the XML file to have the right flux and spectrum
    edit_box_xml(energy, box_flux, z)

    #Do the fitting
    like1 = BinnedAnalysis(obs, 'xmlmodel_fixed_box.xml', optimizer='NEWMINUIT')
    like1.tol=1e-8
    like1obj = pyLike.Minuit(like1.logLike)
    if covar:
        q, c = like1.fit(verbosity=0,optObject=like1obj, covar=covar)
        if get_like:
            print "getting likelihood"
            return q, like1
        else:
            return q
    else:
        q = like1.fit(verbosity=0, tol=1e10, optObject=like1obj, covar=covar, optimizer='NEWMINUIT')
        
        if get_like:
            return q, like1
        else:
            return q
        
        
def get_spectrum(likelihood_obj):
    spectrum = np.zeros((num_ebins-1))
    for source in likelihood_obj.sourceNames():
        spectrum += likelihood_obj._srcCnts(source)
    return spectrum

def plot_spectrum(likelihood_obj, energies, index, window_low, window_high, plot_roi = False):
    
    window_spectrum = get_spectrum(likelihood_obj)
    complete_spectrum = likelihood_obj.nobs
    box_spectrum = likelihood_obj._srcCnts('Box_Component')
    #pedestal = likelihood_obj._srcCnts('Pedestal')
    fig = plt.figure()
    if plot_roi:
        ax = fig.add_subplot(121)
        plt.plot(energies[:-1], window_spectrum, linewidth=2., color='blue', label='Reconstructed spectrum')
        plt.errorbar(energies[:-1], complete_spectrum, xerr=0, yerr=np.sqrt(complete_spectrum), fmt='o', color='black',label='Actual data')
        plt.axvspan(energies[window_low]-0.5*ebin_widths[window_low], energies[window_high]-0.5*ebin_widths[window_high], alpha=0.3, color='black', label='Window')
        plt.plot(energies[:-1], box_spectrum, linewidth=2.0, color='red', label='Box Upper Limit')
        plt.axvline(energies[index], linestyle=':',color='black',linewidth=0.3)
        plt.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([8000.0, 800000])
        ax.set_ylim([0.5, 10**3])
        plt.suptitle('Bin ' + str(index))

        ax = fig.add_subplot(122)
        srcmap_complete = pyfits.open(sourcemap)
        plt.imshow(np.sum(srcmap_complete[0].data[window_low:window_high],axis=0), interpolation='none')
    else:
        ax = fig.add_subplot(111)
        plt.plot(energies[:-1], window_spectrum, linewidth=2., color='blue', label='Reconstructed spectrum')
        plt.errorbar(energies[:-1], complete_spectrum, xerr=0, yerr=np.sqrt(complete_spectrum), fmt='o', color='black',label='Data')
        plt.axvspan(energies[window_low]-0.5*ebin_widths[window_low], energies[window_high]-0.5*ebin_widths[window_high], alpha=0.3, color='black', label='Window')
        plt.plot(energies[:-1], box_spectrum, linewidth=2.0, color='red', ls='-.', label='Box Best Fit')
        #plt.plot(energies[:-1], -1*pedestal, linewidth=2.0, color='green', ls=':', label='Pedestal')
        
        plt.axvline(energies[index], linestyle=':',color='black',linewidth=0.3)
        plt.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([8000.0, 800000])
        ax.set_ylim([10**-2, 10**3])
        plt.suptitle('Bin ' + str(index))
            
    plt.show()
    raw_input('wait for key')

def likelihood_upper_limit3(z):
    
    #Array to hold results of flux upper limit calculation
    num_ebins = 51 #1 more than the number of bins due to the fencepost problem
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
    ebin_widths = np.diff(energies)
    
    sourcemap = '6gev_srcmap_03.fits'#'box_srcmap_artificial_box.fits'#'6gev_srcmap_complete.fits'#
    box_flux = np.zeros((num_ebins-1))
    box_flux_bayesian = np.zeros((num_ebins-1))
    box_flux_frequentist = np.zeros((num_ebins-1))
    
    
    corr = np.zeros((num_ebins-1))
    corr2 = np.zeros((num_ebins-1))

    gll_index = np.zeros((num_ebins-1))
    disk_index = np.zeros((num_ebins-1))
    #reconstructed_spectra = np.zeros((num_ebins-1, num_ebins-1))
    #Loop through upper edge of box
    for index in range(6,48):
        box_width = energies[index]*2.0*np.sqrt(1.0-z)/(1+np.sqrt(1.0-z))
                
        print "Calculating upper limit in bin " + str(index) + " at energy " + str(energies[index])
        #print "bin " + str(np.argmin(np.abs(energies-energy)))
        #window_low, window_high = window(energy, energies)
        
        window_low = index-6
        window_high = index+2

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
            #srcmap_complete[9].data = srcmap_complete[9].data[:window_low+1]
            #m = srcmap_complete[9]

            os.system('rm srcmap_low.fits')
            b.header['DSVAL4'] = str()+':'+str()
            hdulist = pyfits.HDUList([a, srcmap_complete[1], b, c, d, e, f, g, h])#, m])#add in m for pedestal sourcemap
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
            #srcmap_complete[9].data = srcmap_complete[9].data[window_high:]
            #m = srcmap_complete[9]

            os.system('rm srcmap_high.fits')
            hdulist = pyfits.HDUList([a, srcmap_complete[1], b, c, d, e, f, g, h])#, m])
            hdulist.writeto('srcmap_high.fits')
            srcmap_complete.close()

        summedLike = SummedLikelihood()

        if index>6:
            obs_low = BinnedObs(srcMaps='srcmap_low.fits', expCube='6gev_ltcube.fits', binnedExpMap='exposure_low.fits', irfs='CALDB')
            like_low = BinnedAnalysis(obs_low, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
            summedLike.addComponent(like_low)

        if index<48:
            obs_high = BinnedObs(srcMaps='srcmap_high.fits', expCube='6gev_ltcube.fits', binnedExpMap='exposure_high.fits', irfs='CALDB')
            like_high = BinnedAnalysis(obs_high, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
            summedLike.addComponent(like_high)
        
        print "Fitting SummedLikelihood"
        summedLike.ftol = 1e-8
        summedLike.fit(verbosity=3)
        summedLike.writeXml('xmlmodel_free.xml')
        for k in range(len(summedLike.params())):
            summedLike.freeze(k)
        summedLike.writeXml('xmlmodel_fixed.xml')
        
        print "Fitting all data"
        
        calculation = 'complete'
        obs_complete = BinnedObs(srcMaps=sourcemap, expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='CALDB')
        edit_box_xml(100000.0,1e-15,0.0)
        like = BinnedAnalysis(obs_complete, 'xmlmodel_fixed_box.xml', optimizer='MINUIT')
        like.tol=1e-8
        like_obj = pyLike.Minuit(like.logLike)
        like.fit(verbosity=3,optObject=like_obj)
        like.writeXml('xmlmodel_fixed_box.xml')
        """
        #Flucuate the window data
        f = pyfits.open(sourcemap)
        poisson_data = np.zeros((len(range(max(window_low,0), min(window_high, 49))), 50, 50))
        q = 0
        for bin in range(max(window_low,0), min(window_high,49)):
            for source in like.sourceNames():
                for j in range(3,9):#Change to 10 for pedestal sourcemap
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
        like = BinnedAnalysis(obs_poisson, 'xmlmodel_fixed_box.xml', optimizer='NEWMINUIT')
        like.tol=1e-8
        like_obj = pyLike.Minuit(like.logLike)
        like.fit(verbosity=0,optObject=like_obj)
        """
        if calculation == 'complete':
             obs_calculation=obs_complete
        else:
            obs_calculation= obs_poisson
        null_likelihood = -1.0*loglikelihood2(energies[index], 1.0e-15, obs_calculation, z)
        crit_chi2 = 2.706
        
        #Likelihood profiles
        print "Likelihoods:" 
        print loglikelihood2(energies[index], 1.0e-15, obs_calculation, z)
        print loglikelihood3(energies[index], 1.0e-15, obs_calculation, z)
        raw_input('wait for key')
        """"
        fig = plt.figure()
        ax = fig.add_subplot(121)
        
        likelihood_profile = np.zeros((50))
        profile_fluxes = 10**np.linspace(-15, -10, 50)
        for w in range(50):
            likelihood_profile[w] = -1.0*loglikelihood2(energies[index], profile_fluxes[w], obs_calculation, z)
        ax.plot(profile_fluxes, likelihood_profile)
        ax.axhline(null_likelihood+crit_chi2, linewidth=0.5, color='black', linestyle='--')
        plt.xscale('log')
        
        ax2 = fig.add_subplot(122)
        null_likelihood2 = loglikelihood3(energies[index], 1.0e-15, obs_calculation, z)
        
        likelihood_profile2 = np.zeros((50))
        profile_fluxes = 10**np.linspace(-15, -10, 50)
        for w in range(50):
            likelihood_profile2[w] = loglikelihood3(energies[index], profile_fluxes[w], obs_calculation, z)
        ax2.plot(profile_fluxes, likelihood_profile2)
        ax2.axhline(null_likelihood2+crit_chi2, linewidth=0.5, color='black', linestyle='--')
        plt.xscale('log')
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.show()
        """
        
        print "Finding Upper Limit..."
        null_likelihood = loglikelihood2(energies[index], 1.0e-15, obs_calculation, z)
        delta_loglike = 0.0
        #Is it delta log like from max likelihood or null hypothesis?
        
        #increase box flux until likelihood > 2sigma over null likelihood
        box_flux[index] = 3.e-15
        while delta_loglike <crit_chi2:
            box_flux[index]*=3.0                
            delta_loglike = 2.0*(loglikelihood2(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
            print "stage 1"
            print "delta loglike = " + str(delta_loglike)
            print "flux = " + str(box_flux[index]) + " likelihood = " + str(loglikelihood2(energies[index], box_flux[index], obs_calculation, z))
            
            
        box_flux[index]*=1.0/3.0
        delta_loglike = 2.0*(loglikelihood2(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
        print "Delta loglike = " + str(delta_loglike)
        while delta_loglike <crit_chi2:
            box_flux[index]*=1.5
            delta_loglike = 2.0*(loglikelihood2(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
            print "stage 2"
            print "delta loglike = " + str(delta_loglike)
            print "flux = " + str(box_flux[index]) + " likelihood = " + str(loglikelihood2(energies[index], box_flux[index], obs_calculation, z))
            
            
        box_flux[index]*=1.0/1.5
        delta_loglike = 2.0*(loglikelihood2(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
        
        while delta_loglike <crit_chi2:
            box_flux[index]*=1.03
            delta_loglike = 2.0*(loglikelihood2(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
            print "stage 3"
            print "delta loglike = " + str(delta_loglike)
            print "flux = " + str(box_flux[index]) + " likelihood = " + str(loglikelihood2(energies[index], box_flux[index], obs_calculation, z))
            
            
        box_flux[index]*= 1.0/1.03
        delta_loglike = 2.0*(loglikelihood2(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
        
        print "delta log like = " + str(delta_loglike)
        
        high_like = BinnedAnalysis(obs_calculation, 'xmlmodel_fixed_box.xml', optimizer='DRMNFB')
        #plot_spectrum(high_like, energies, index, window_low, window_high, plot_roi = False)
    
        
        calc_cov = True
        if calc_cov:
            like1 = BinnedAnalysis(obs_calculation, 'xmlmodel_fixed_box.xml', optimizer='DRMNFB')
            
            like1.thaw(like1.par_index('Disk Component','Index'))
            like1.thaw(like1.par_index('Disk Component','Prefactor'))
            like1.thaw(like1.par_index('Box_Component','Normalization'))
            like1.tol=1e-5
            like1obj = pyLike.Minuit(like1.logLike)
            like1.fit(verbosity=0,optObject=like1obj, covar=False)

            like1.writeXml('xmlmodel_fixed_box.xml')
            
            like2 = BinnedAnalysis(obs_calculation, 'xmlmodel_fixed_box.xml', optimizer='NewMinuit')
            like2.tol=1e-8
            like2obj = pyLike.Minuit(like1.logLike)
            like2.fit(verbosity=0,optObject=like1obj, covar=True)
            
            
            #ul = UpperLimit(like1,'Box Component')
            #ul.compute(emin=100.0,emax=500000, delta=3.91)
            
            #box_flux_bayesian[index] = float(ul.bayesianUL()[0])
            #box_flux_frequentist[index] = float(ul.results[0].value)            
            print like2.covariance
            cov = like2.covariance
            corr[index] = cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])
            corr2[index] = cov[0][2]/np.sqrt(cov[0][0]*cov[2][2])
            print "Correlations:"
            print corr[index]
            print corr2[index]
            file = open('correlation_results.pk1', 'wb')
            pickle.dump([corr, corr2],file)
            file.close()
            
            #if like2obj.getRetCode()!=0:
    
    if calc_cov:
        file = open('correlation_results.pk1', 'wb')
        pickle.dump([corr, corr2, box_flux_bayesian, box_flux_frequentist],file)
        file.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(energies[np.nonzero(corr)],corr[np.nonzero(corr)], color='blue', label='Box vs GC Prefactor')
        ax.plot(energies[np.nonzero(corr2)],corr2[np.nonzero(corr2)], color='red', label='Box vs GC index')
        plt.ylim([-1.0, 1.0])
        
        plt.legend()
        plt.xscale('log')
    
        
        plt.show()
    
    return box_flux

def consolidate_brazil_lines(filename):
    file = open(filename,'rb')
    g = pickle.load(file)
    file.close()
    brazil_dict = np.zeros((6100,num_ebins-1))
    i = 0
    for entry in g:
        brazil_dict[i,:] = entry
        i += 1
    print str(i) + " MC events"
    return brazil_dict



def make_ul_plot(ul, brazil_dict, plot_type, plt_title):
    mc_limits = brazil_dict[np.nonzero(brazil_dict[:,10])]
    trials = len(mc_limits)
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
    lower_95 = savgol_filter(lower_95[6:48],11,1)
    upper_95 = savgol_filter(upper_95[6:48],11,1)
    lower_68 = savgol_filter(lower_68[6:48],11,1)
    upper_68 = savgol_filter(upper_68[6:48],11,1)
    
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111)
    if plot_type=='UL':
        #Plotting uppper limit
        #ax.plot(energies[:-1],median,color='black',linewidth=1,linestyle='--', label='Median MC')
        ax.fill_between(energies[:-1][6:48], lower_95, upper_95, color='yellow', label='95\% MC Containment')
        ax.fill_between(energies[:-1][6:48], lower_68, upper_68, color='#63ff00',label='68\% MC Containment')
        ax.plot(energies[:-1][6:48],ul[6:48], marker='.', markersize=13.0,color='black',linewidth=2, label='95\% Confidence Upper Limit')
        
        chris_e = np.array([1129.4626874999999, 1421.9093124999999, 1790.0778125000002, 2253.5744374999999, 2837.0821249999999, 3571.6747500000001, 4496.4721250000002, 5660.723, 7126.4279999999999, 8971.6412500000006, 11294.627, 14219.093000000001, 17900.777999999998, 22535.743999999999, 28370.82, 35716.745999999999, 44964.720000000001, 56607.230000000003, 71264.279999999999, 89716.412000000011],'d')        
        chris_list = [(1,4.77e-10),(2,4.73e-10),(3,1.46e-09),(5,1.00e-08),(6,1.02e-08),(8,4.89e-09),(9,3.02e-09),(7,7.57e-09),(10,1.23e-09),(4,7.95e-09),(16,8.06e-11),(15,1.20e-10),(12,3.79e-10),(17,6.02e-11),(19,4.67e-11),(11,6.22e-10),(14,1.62e-10),(18,5.30e-11),(13,2.43e-10),(20,4.59e-11)]
        chris_ul = np.zeros((20))
        for entry in chris_list:
            chris_ul[entry[0]-1] = entry[1]
        #ax.plot(chris_e, chris_ul, marker='None', linestyle='--', color='black',linewidth=1.0, label='95\% Confidence Upper Limit (Pass 7 Data)')

        ax.errorbar(np.array([1e5]), np.array([2*10**-10]), xerr=0, yerr=np.array([1.4*8*10**-10-8*10**-10]), color='blue', markersize=7, fmt='o', label='Injected Box')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel('Flux Upper Limit [ph s$^{-1}$ cm$^{-2}$]')
        plt.xlabel('Energy [MeV]')
        plt.legend()
        
        ax.set_xlim([energies[6], energies[47]])
        ax.set_ylim([2*10**-12, 2*10**-9])
        plt.savefig('plots/'+str(plt_title),bbox_inches='tight')
    if plot_type=='SIG':
        significances = np.zeros((len(ul)))
        for i in range(6,48):
            p_value = float(np.argmin(np.abs(mc_limits[:,i]-ul[i])))/float(len(mc_limits[:,i]))
            if p_value<0.5:
                significances[i] = -1.0*sigma_given_p(p_value)
            else:
                significances[i] = sigma_given_p(1.0-p_value)
        #energies = np.delete(energies, 41)
        ax.fill_between(energies[:-1], -2.0, 2.0, color='yellow', label='95\% Containment')
        ax.fill_between(energies[:-1], -1.0, 1.0, color='#63ff00', label='68\% Containment')
        ax.axhline(0.0, color='black', linestyle='--', linewidth=0.5)
        plt.plot(energies[:-1], significances, color='black', marker='.', linewidth=2, markersize=13.0, label='Data')
        plt.ylabel('Significance [$\sigma$]')
        ax.set_xscale('log')
        ax.set_xlim([energies[6], energies[47]])
        ax.set_ylim([-4.0, 4.0])
        plt.xlabel('Energy [MeV]')
        plt.legend()
        plt.savefig('plots/'+str(plt_title),bbox_inches='tight')



def plotter(curves):
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111)
    g=0.25
    for curve, z in zip(curves,[0.0, 0.44, 0.85, 0.99]):
        ax.plot(energies[:-1],curve, marker='.', markersize=13.0,color=cm.rainbow(z**2),linewidth=2, label='$\zeta='+str(z)+'$')
    
    plt.gca().set_facecolor('white')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Flux Upper Limit [ph s$^{-1}$ cm$^{-2}$]')
    plt.xlabel('Energy [MeV]')
    plt.legend()

    ax.set_xlim([energies[6], 600000])
    ax.set_ylim([2*10**-12, 8*10**-10])
    plt.savefig('plots/zeta_comparison.pdf',bbox_inches='tight')
    plt.show()
    
    
#make_ul_plot(box_flux, brazil_dict, plot_type='UL')
def main():
    #z0 = likelihood_upper_limit3(0.0)
    #file = open('z0.pk1','wb')
    #pickle.dump(z0,file)
    #file.close()
    
    file = open('z0.pk1','rb')
    z0 = pickle.load(file)
    file.close()
    
    #z44 = likelihood_upper_limit3(0.44)
    #file = open('z44.pk1','wb')
    #pickle.dump(z44,file)
    #file.close()
    
    file = open('z44.pk1','rb')
    z44 = pickle.load(file)
    file.close()
    
    #z85 = likelihood_upper_limit3(0.85)
    #file = open('z85.pk1','wb')
    #pickle.dump(z85,file)
    #file.close()
    
    file = open('z85.pk1','rb')
    z85 = pickle.load(file)
    file.close()
    
    #z99 = likelihood_upper_limit3(0.9999)
    #file = open('z99.pk1','wb')
    #pickle.dump(z99,file)
    #file.close()
    
    file = open('z99.pk1','rb')
    z99 = pickle.load(file)
    file.close()
    
    z_artificial_box = likelihood_upper_limit3(0.0)
    file = open('z_artificial.pk1','wb')
    pickle.dump(z_artificial_box,file)
    file.close()
    
    file = open('z_artificial.pk1','rb')
    z_artificial_box = pickle.load(file)
    file.close()

    
    #plotter([z0, z44, z85, z99])
    #brazil_dict = consolidate_brazil_lines('wide_box_brazil.pk1')
    #make_ul_plot(z44,brazil_dict,'UL','brazil_wide_box.pdf')
    #make_ul_plot(z44,brazil_dict,'SIG', 'significance_wide_box.pdf')

    #brazil_dict = consolidate_brazil_lines('narrow_box_brazil.pk1')
    #make_ul_plot(z99,brazil_dict,'UL','brazil_narrow_box.pdf')
    #make_ul_plot(z99,brazil_dict,'SIG', 'significance_narrow_box.pdf')
    
    brazil_dict = consolidate_brazil_lines('artificial_box_brazil.pk1')
    make_ul_plot(z_artificial_box, brazil_dict,'UL','artificial_box.pdf')
    make_ul_plot(z_artificial_box, brazil_dict,'SIG', 'significance_artificial_box.pdf')



if __name__ == '__main__':
    main()