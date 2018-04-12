print("Loading required packages...")
#Math packages, NumPy
import math
import numpy as np
from BinnedAnalysis import *

#pfrom scipy.special import gammainc, erf, gamma
from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
import scipy
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from operator import add
import pickle

#Fermi Science Tools
#from SummedLikelihood import *

#Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.pyplot import rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Wedge, Circle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord

print("Done!")

gc_l = 359.94425518526566
gc_b = -0.04633599860905694
gc_ra = 266.417
gc_dec = -29.0079

def setup_plot_env():
    #Set up figure
    #Plotting parameters
    fig_width = 8   # width in inches
    fig_height = 8  # height in inches
    fig_size =  [fig_width, fig_height]
    rcParams['font.family'] = 'serif'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelsize'] = 24
    rcParams['font.size'] = 26
    rcParams['axes.titlesize'] =16
    rcParams['legend.fontsize'] = 16
    rcParams['xtick.labelsize'] =28
    rcParams['ytick.labelsize'] =28
    rcParams['figure.figsize'] = fig_size
    rcParams['xtick.major.size'] = 14
    rcParams['ytick.major.size'] = 14
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

#A derived class from the Science Tools BinnedAnalysis class
#Adds various methods which make life easier
class AnalyticAnalysis(BinnedAnalysis):

    #Methods to loop over the sources & freeze them all
    def freeze_all_sources(self):
        for k in range(len(self.params())):
            self.freeze(k)

    def free_all_sources(self):
        for k in range(len(self.params())):
            if self.params()[k].parameter.getName()!='Scale' and self.params()[k].parameter.getName()!='Eb':
                self.thaw(k)

    def edit_parameter(self, source_name, parameter_name, new_value):
        self[source_name]['Spectrum'][parameter_name] = new_value
        #self.fit(verbosity=0)

    #The fit & optimize methods of BinnedAnalysis already return the minus loglikelihood, but it's unclear exactly how it is calculated
    #Here it's just calculated from the definition
    def loglikelihood(self):
        #The sourcemap file is needed to calculate the model.
        sourcemap = self.binnedData.srcMaps
        f = fits.open(sourcemap)

        #Given the results of the fit, calculate the model
        model_data = np.zeros(f[0].shape)
        for source in self.sourceNames():
            the_index = f.index_of(source)
            model_data += self._srcCnts(source)[:, None, None]*f[the_index].data[:-1, :, :]/np.sum(np.sum(f[the_index].data, axis=2), axis=1)[:-1, None, None]
        actual_data = np.array(self.binnedData.countsMap.data()).reshape(f[0].shape)
        #Likelihood value is a product of Poisson factors
        likelihood = np.sum(np.sum(np.sum(np.log(model_data**actual_data*np.exp(-1.0*model_data)/factorial(actual_data)))))

        #Return minus log-likelihood- a function to be minimized
        return -1.0*likelihood

    #Calculate the covariance matrix between two sources with a simple difference method. Works on boundary values as well
    def calculateCovarianceMatrix(self, source_A, parameter_A, source_B, parameter_B):
        current_val_a = self[source_A]['Spectrum'][parameter_A]
        current_val_b = self[source_B]['Spectrum'][parameter_B]

        #First, calculate Hessian to a) confirm we are at a minimum and
        #b) covariance matrix is the inverse of the hessian
        num_parameters = 2
        hessian = np.zeros((num_parameters, num_parameters))

        dx_a = 100.0
        dx_b = np.abs(0.1*current_val_b)

        for a in range(num_parameters):
            for b in range(num_parameters):
                print(a, b)
                if a == b:
                    if a == 0:
                        print(self[source_A]['Spectrum'][parameter_A])
                        self.edit_parameter(source_A, parameter_A, current_val_a+2*dx_a)
                        print(self[source_A]['Spectrum'][parameter_A])
                        yiplus1 = self.loglikelihood()

                        self.edit_parameter(source_A, parameter_A, current_val_a+dx_a)
                        yi = self.loglikelihood()

                        self.edit_parameter(source_A, parameter_A, current_val_a)
                        yiminus1 = self.loglikelihood()
                        print(y1plus1, yi, yiminus1)
                        hessian[a,b] = (yiplus1-2.0*yi+yiminus1)/(dx_a**2)
                    if a == 1:
                        self.edit_parameter(source_B, parameter_B, current_val_b+dx_b)
                        yiplus1 = self.loglikelihood()

                        self.edit_parameter(source_B, parameter_B, current_val_b)
                        yi = self.loglikelihood()

                        self.edit_parameter(source_B, parameter_B, current_val_b-dx_b)
                        yiminus1 = self.loglikelihood()

                        hessian[a,b] = (yiplus1-2.0*yi+yiminus1)/(dx_b**2)

                else:
                    z = np.zeros((2,2))
                    self.edit_parameter(source_A, parameter_A, current_val_a+2*dx_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b+dx_b)
                    z[0,0] = self.loglikelihood()

                    self.edit_parameter(source_A, parameter_A, current_val_a+2*dx_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b-dx_b)
                    z[0,1] = self.loglikelihood()

                    self.edit_parameter(source_A, parameter_A, current_val_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b+dx_b)
                    z[1,0] = self.loglikelihood()

                    self.edit_parameter(source_A, parameter_A, current_val_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b-dx_b)
                    z[1,1] = self.loglikelihood()
                    hessian[a,b] = (z[1,1]-z[0,1]-z[1,0]+z[0,0])/(4.0*dx_a*dx_b)

        if np.sign(np.linalg.det(hessian)) == -1.0:
            print("WARNING: Determinant of Hessian matrix is negative- you are probably not at an extremum of the function you are trying to optimize")
            return 0
        print("hessian = " + str(hessian))

        #Covariance matrix is the inverse of the hessian
        covariance = np.linalg.inv(hessian)
        self.covariance = covariance

    def getCovarianceMatrix(self):
        return self.covariance

    def calculateCorrelationMatrix(self, covariance):
        correlation = np.zeros((covariance.shape))
        for a in covariance.shape[0]:
            for b in covariance.shape[1]:
                correlation[a,b] = covariance[a][b]/np.sqrt(float(covariance[a][a])*float(covariance[b][b]))
        self.correlation = correlation

    def getCorrelationMatrix(self):
        return self.correlation

    def getSpectrum(self):
        spectrum = np.zeros((num_ebins-1))
        for source in self.sourceNames():
            spectrum += self._srcCnts(source)
        return spectrum

    #Make a spectral plot showing fitted data
    def spectralPlot(self, filename):
        box_spectrum = self._srcCnts('Box_Component')
        window_spectrum = self.getSpectrum()
        complete_spectrum = self.nobs
        #pedestal = likelihood_obj._srcCnts('Pedestal')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(energies[:-1], window_spectrum, linewidth=2., color='blue', label='Reconstructed spectrum')
        plt.errorbar(energies[:-1], complete_spectrum, xerr=0, yerr=np.sqrt(complete_spectrum), fmt='o', color='black',label='Data')
        plt.plot(energies[:-1], box_spectrum, linewidth=2.0, color='red', ls='-.', label='Box Best Fit')
        plt.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([8000.0, 800000])
        ax.set_ylim([10**-2, 10**3])
        plt.title(filename)
        plt.show()
        #plt.savefig(filename, bbox_inches='tight')

    #Make a plot of the data/model/residuals
    def roiPlot(self):
        sourcemap = self.binnedData.srcMaps
        f = fits.open(sourcemap)

        image_data = fits.getdata('6gev_image.fits')
        filename = get_pkg_data_filename('6gev_image.fits')
        hdu = fits.open(filename)[0]
        wcs = WCS(hdu.header)

        #Given the results of the fit, calculate the model
        model_data = np.zeros(f[0].shape)
        for source in self.sourceNames():
            the_index = f.index_of(source)
            model_data += self._srcCnts(source)[:, None, None]*f[the_index].data[:-1, :, :]/np.sum(np.sum(f[the_index].data, axis=2), axis=1)[:-1, None, None]
        actual_data = np.array(self.binnedData.countsMap.data()).reshape(f[0].shape)

        fig = plt.figure(figsize=[14,6])

        ax = fig.add_subplot(131, projection=wcs)
        ax=plt.gca()

        c = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
        ax.add_patch(c)
        mappable=plt.imshow(np.sum(actual_data, axis=0),cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='gaussian')#
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        plt.title('Data')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(mappable, cax=cax, label='Counts per pixel')

        ax2=fig.add_subplot(132, projection=wcs)
        ax2 = plt.gca()
        c2 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax2.get_transform('galactic'))
        ax2.add_patch(c2)
        mappable2 = plt.imshow(np.sum(model_data, axis=0), cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='gaussian')
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        plt.title('Model')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cb2 = plt.colorbar(mappable2, cax=cax2, label='Counts per pixel')

        ax3=fig.add_subplot(133, projection=wcs)
        ax3 = plt.gca()
        c3 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax3.get_transform('galactic'))
        ax3.add_patch(c3)
        mappable3 = plt.imshow(np.sum(actual_data, axis=0)-np.sum(model_data, axis=0), cmap='seismic',origin='lower', vmin=-20, vmax=20, interpolation='gaussian')#
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        plt.title('Residuals')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cb3 = plt.colorbar(mappable3, cax=cax3, label='Counts per pixel')
        fig.tight_layout()

        plt.show()

    def likelihoodMap(self, source_A, parameter_A, source_B, parameter_B, res = 10, verbose=False, generate_plot=False):
        print("Making likelihood map...")
        like_mat = np.zeros((res,res))
        l = 0
        k = 0

        #Find a good range of parameters from the Hessian
        cutoff_delta_loglike = 10.0
        xspace = np.linspace(max(0,current_val_a-np.sqrt(2*cutoff_delta_loglike/hessian[0,0])), current_val_a+np.sqrt(2*cutoff_delta_loglike/hessian[0,0]), res)
        yspace = np.linspace(current_val_b-np.sqrt(2*cutoff_delta_loglike/hessian[1,1]), current_val_b+np.sqrt(2*cutoff_delta_loglike/hessian[1,1]), res)
        for xval in xspace:
            l = 0
            for yval in yspace:
                self.edit_parameter(source_A, parameter_A, xval)
                self.edit_parameter(source_B, parameter_B, yval)

                like_mat[k, l] = self.loglikelihood()
                if verbose:
                    print(source_B + " " + parameter_B + " = " + str(self[source_B]['Spectrum'][parameter_B]))
                    print(source_A + " " + parameter_A + " = " + str(self[source_A]['Spectrum'][parameter_A]))
                l += 1
            k += 1

        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        mappable = plt.imshow(like_mat, origin='lower', extent=[min(yspace), max(yspace), min(xspace), max(xspace)], aspect='auto')
        plt.colorbar(mappable, label='LogLikelihood')
        plt.xlabel(source_B + " " + parameter_B)
        plt.ylabel(source_A + " " + parameter_A)

        self.edit_parameter(source_A, parameter_A, current_val_a)
        self.edit_parameter(source_B, parameter_B, current_val_b)

        fig.tight_layout()
        plt.show()


def roiPlot():
    sourcemap = '6gev_srcmap_03.fits'
    f = fits.open(sourcemap)

    image_data = fits.getdata('6gev_image.fits')
    filename = get_pkg_data_filename('6gev_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    file = open('image_data.pk1','rb')
    [actual_data, model_data] = pickle.load(file)
    file.close()

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection=wcs)
    ax=plt.gca()
    c = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)
    mappable=plt.imshow(np.sum(actual_data, axis=0),cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='gaussian')#
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    cb = plt.colorbar(mappable, fraction=0.0458, pad = 0.01, label='Counts per pixel')
    plt.tight_layout()
    plt.savefig('/Users/christian/ROI_6gev.pdf',bbox_inches='tight')

    fig = plt.figure(figsize=[10,8])
    ax2=fig.add_subplot(111, projection=wcs)
    ax2 = plt.gca()
    c2 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax2.get_transform('galactic'))
    ax2.add_patch(c2)
    mappable2 = plt.imshow(np.sum(model_data, axis=0), cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='gaussian')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    cb2 = plt.colorbar(mappable2,fraction=0.0458, pad = 0.01, label='Counts per pixel')
    plt.tight_layout()
    plt.savefig('/Users/christian/ROI_model_6gev.pdf',bbox_inches='tight')

    fig = plt.figure(figsize=[10,8])
    ax3=fig.add_subplot(111, projection=wcs)
    ax3 = plt.gca()
    c3 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax3.get_transform('galactic'))
    ax3.add_patch(c3)
    mappable3 = plt.imshow(np.sum(actual_data, axis=0)-np.sum(model_data, axis=0), cmap='seismic',origin='lower', vmin=-20, vmax=20, interpolation='gaussian')#
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    cb3 = plt.colorbar(mappable3, fraction=0.0458, pad = 0.01, label='Counts per pixel')
    plt.tight_layout()
    plt.savefig('/Users/christian/residmap_6gev.pdf',bbox_inches='tight')

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
    print("Upper limit is " + str(s))
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
        print("Integral must be performed with equal-sized arrays!")
        print("Length of x is " + str(len(x)) + " Length of g is " + str(len(g)))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))

def update_box_spectrum(energy, zeta):
    box_width = energy*2.0*np.sqrt(1.0-zeta)/(1+np.sqrt(1.0-zeta))
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
    integrated_box_flux = get_integral(x_fine_grid, convolved_pure_box)
    for i in range(len(x_fine_grid)):
        spectrum_file.write(str(x_fine_grid[i])+" " + str(max(convolved_pure_box[i]/integrated_box_flux, 10.**-35))+"\n")
    spectrum_file.close()

    return box_width, integrated_box_flux

def quadratic(x, a, b, c):
    return a*x**2+b*x+c

num_ebins = 51 #1 more than the number of bins due to the fencepost problem
energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
ebin_widths = np.diff(energies)

def likelihood_upper_limit3(zeta, sourcemap, poisson=False):
    scale_factor = 1e-15
    crit_chi2 = 2.71 #For 95% confidence one-sided upper limit with 1 degree of freedom

    #Arrays to store results in
    box_flux = np.zeros((num_ebins-1, 3))
    correlations = np.zeros((num_ebins, 2))
    #sourcemap = '6gev_srcmap_03.fits'

    #Instantiate the analysis objects
    print("Loading data...")
    obs_complete = BinnedObs(srcMaps=sourcemap, expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='P8R2_SOURCE_V6')
    print("obs loaded")
    like = BinnedAnalysis(obs_complete, 'xmlmodel.xml', optimizer='MINUIT')

    #For MC study i.e. making Brazil plot bands
    if poisson:
        print("Calculating fluctuations...")
        #Poisson fluctuations of the data
        f = fits.open(sourcemap)
        #Given the results of the fit, calculate the model
        model_data = np.zeros(f[0].shape)
        for source in like.sourceNames():
            the_index = f.index_of(source)
            model_data += like._srcCnts(source)[:, None, None]*f[the_index].data[:-1, :, :]/np.sum(np.sum(f[the_index].data, axis=2), axis=1)[:-1, None, None]

        #Introduce Poisson fluctuations on top of the model
        poisson_data = np.random.poisson(model_data)

        #Write to a new sourcemap
        f[0].data = poisson_data
        f.writeto('box_srcmap_poisson.fits')
        f.close()
        obs_poisson = BinnedObs(srcMaps='box_srcmap_poisson.fits', expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='P8R2_SOURCE_V6')

    #Loop through upper edge of box
    for index in range(34,48):
        print("Evaluating box with upper edge " + str(energies[index]) + " MeV in bin " + str(index))
        #Update the box spectrum
        box_width, integrated_box_flux = update_box_spectrum(energies[index], zeta)
        if poisson:
            like = AnalyticAnalysis(obs_poisson, 'xmlmodel.xml', optimizer='NewMinuit')
            print("Fitting...")
            likeobj = pyLike.Minuit(like.logLike)
            like.free_all_sources()
            like.tol=1e-9
            loglike = like.fit(verbosity=0,optObject=likeobj, covar=False)
            print("Return code: " + str(likeobj.getRetCode()))
            print("loglike = " + str(loglike))
            while loglike> 18000.0:
                loglike = like.fit(verbosity=0,optObject=likeobj, covar=False)
                print("Return code: " + str(likeobj.getRetCode()))
                print("loglike = " + str(loglike))
            like.freeze_all_sources()
        like.spectralPlot('plots/'+str(index))
        input('wait for key')
        #like.writeXml('xmlmodel.xml')

        like.thaw(like.par_index('Disk Component','Prefactor'))
        print("Scanning likelihood...")
        #Scan the likelihood profile to find best-fit value and upper limit
        minx = 0.0
        maxx = 10**-9/scale_factor
        x_range, l_range = like.scan('Box_Component', 'Normalization', minx, maxx, 200)
        #maxx = x_range[max(1, np.argmin(np.abs(l_range-(crit_chi2+min(l_range)))))]*2.0

        #Results
        box_flux[index, 0] = x_range[np.argmin(np.abs(l_range-(crit_chi2+min(l_range))))]*scale_factor #Flux upper limit
        box_flux[index, 1] = x_range[np.argmin(l_range)]*scale_factor #Best fit flux
        box_flux[index, 2] = l_range[np.argmin(l_range)]-l_range[0] #Delta log like for best-fit flux
        print("Best-fit box = " + str(box_flux[index,1]))
        if box_flux[index,1]>0.0:
            print("Significance = " + str(sigma_given_p(pvalue_given_chi2(-2.0*box_flux[index,2], 1))) + " sigma")
        print("Box upper limit = " + str(box_flux[index, 0]))

        print("Calculating correlation matrix...")
        #Correlations between the sources
        #like.calculateCovarianceMatrix('Box_Component', 'Normalization', 'Disk Component', 'Prefactor')
        #like.calculateCorrelationMatrix()
        #correlations[index, 0] = like.correlation[0,1]

        #like.calculateCovarianceMatrix('Box_Component', 'Normalization', 'Disk Component', 'Index')
        #like.calculateCorrelationMatrix()
        #correlations[index, 1] = like.correlation[0,1]

        print("Correlation with GC Prefactor: " + str(correlations[index,0]))
        print("Correlation with GC Index: " + str(correlations[index,0]))


    return box_flux, correlations

def consolidate_brazil_lines(filename):
    file = open(filename,'rb')
    g = pickle.load(file)
    file.close()
    brazil_dict = np.zeros((6100,num_ebins-1))
    i = 0
    for entry in g:
        brazil_dict[i,:] = entry
        i += 1
    print(str(i) + " MC events")
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
    median = savgol_filter(median[6:48], 11, 1)
    print("median expected = " + str(median[6:48]))
    print("energies = " + str(energies[:-1][6:48]))
    print("Data = "  + str(ul[6:48]))
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111)
    if plot_type=='UL':
        #Plotting uppper limit
        #ax.plot(energies[:-1],median,color='black',linewidth=1,linestyle='--', label='Median MC')
        ax.fill_between(energies[:-1][6:48], lower_95, upper_95, color='yellow', label='95\% Containment')
        ax.fill_between(energies[:-1][6:48], lower_68, upper_68, color='#63ff00',label='68\% Containment')
        ax.plot(energies[:-1][6:48],ul[6:48], marker='.', markersize=13.0,color='black',linewidth=2, label='95\% Confidence Upper Limit')
        ax.plot(energies[:-1][6:48], median, linestyle='--', linewidth=0.5, color='black', label='Median Expected')
        chris_e = np.array([1129.4626874999999, 1421.9093124999999, 1790.0778125000002, 2253.5744374999999, 2837.0821249999999, 3571.6747500000001, 4496.4721250000002, 5660.723, 7126.4279999999999, 8971.6412500000006, 11294.627, 14219.093000000001, 17900.777999999998, 22535.743999999999, 28370.82, 35716.745999999999, 44964.720000000001, 56607.230000000003, 71264.279999999999, 89716.412000000011],'d')
        chris_list = [(1,4.77e-10),(2,4.73e-10),(3,1.46e-09),(5,1.00e-08),(6,1.02e-08),(8,4.89e-09),(9,3.02e-09),(7,7.57e-09),(10,1.23e-09),(4,7.95e-09),(16,8.06e-11),(15,1.20e-10),(12,3.79e-10),(17,6.02e-11),(19,4.67e-11),(11,6.22e-10),(14,1.62e-10),(18,5.30e-11),(13,2.43e-10),(20,4.59e-11)]
        chris_ul = np.zeros((20))
        for entry in chris_list:
            chris_ul[entry[0]-1] = entry[1]
        #ax.plot(chris_e, chris_ul, marker='None', linestyle='--', color='black',linewidth=1.0, label='95\% Confidence Upper Limit (Pass 7 Data)')

        #To show a dot where an injected signal lives
        #ax.errorbar(np.array([1e5]), np.array([3.0*10**-10]), xerr=0, yerr=0.0, color='blue', markersize=10, fmt='o', label='Injected Signal')

        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel('Flux Upper Limit [ph s$^{-1}$ cm$^{-2}$]')
        plt.xlabel('Energy [MeV]')
        plt.legend(loc=3)

        ax.set_xlim([energies[6], energies[47]])
        ax.set_ylim([2*10**-12, 2*10**-9])
        plt.savefig('plots/'+str(plt_title),bbox_inches='tight')
        plt.show()

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

def correlationPlot(correlation1, correlation2):
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(111)
    plt.plot(energies[:-1][6:48], correlation1[:,0][6:48], linewidth=2.0, color='blue', label='$\zeta=0.44$ Signal vs GC Prefactor')
    plt.plot(energies[:-1][6:48], correlation2[:,0][6:48], linewidth=2.0, color='red', label='$\zeta=0.9999$ Signal vs GC Prefactor')
    plt.axhline(0.0, color='black', linestyle='--', linewidth=0.5)
    plt.xscale('log')
    plt.ylim([-1.0, 0.1])
    plt.xlim([energies[6], energies[47]])
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Box Upper Edge [MeV]')
    plt.legend()
    plt.savefig('plots/correlation_coefficients.pdf',bbox_inches='tight')
    plt.show()


def multiPlotter(curves):
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

def theoryPlotter():
    ad_gammac1_z99 = [[19.6237, 4.57491*10**-6], [21.6411, 7.505*10**-6], [23.866, 9.17377*10**-6], [26.3195, 0.0000118648], [29.0253, 0.0000171208], [32.0092, 0.0000188892], [35.2999, 0.0000101674], [38.9289, 7.94083*10**-6], [42.931, -3.10589*10**-6], [47.3446, -4.73139*10**-6], [52.2118, 0.0000130573], [57.5795, 0.0000164174], [63.4989, 0.0000131265], [70.027, 0.0000130879], [77.2261, 0.0000262152], [85.1653, 0.0000570469], [93.9208, 0.0000740881], [103.576, 0.0000414739], [114.224, 0.0000241062], [125.967, 0.0000303306], [138.917, 0.000063084], [153.199, 0.00011705], [168.948, 0.000181654], [186.317, 0.000178264], [205.472, 0.000191036], [226.595, 0.000178923], [249.89, 0.000116762], [275.58, 0.0000671643], [303.911, 0.0000579031], [335.155, 0.000125408], [369.611, 0.000351548], [407.608, 0.000910204], [449.513, 0.00188837], [495.725, 0.00215645], [546.688, 0.00135476], [602.89, 0.000651263], [664.871, 0.000581901], [733.223, 0.000754332], [808.602, 0.000784346], [891.73, 0.000791298], [983.405, 0.00102506], [1084.5, 0.00115408]]

    gammac1_gammasp18_z99 = [[19.6237, 1.67415], [21.6411, 4.10128], [23.866, 5.25415], [26.3195, 7.31753], [29.0253, 12.0055], [32.0092, 13.35], [35.2999, 5.35204], [38.9289, 1.74911], [42.931, 1.38417], [47.3446, 2.89916], [52.2118, 6.69735], [57.5795, 8.86803], [63.4989, 6.41571], [70.027, 5.91757], [77.2261, 15.4644], [85.1653, 46.0979], [93.9208, 65.0021], [103.576, 26.8298], [114.224, 12.196], [125.967, 16.1324], [138.917, 44.1432], [153.199, 104.859], [168.948, 189.835], [186.317, 178.596], [205.472, 190.311], [226.595, 167.165], [249.89, 86.3939], [275.58, 37.588], [303.911, 29.7895], [335.155, 85.5436], [369.611, 367.407], [407.608, 1275.23], [449.513, 3002.27], [495.725, 3451.53], [546.688, 1970.76], [602.89, 740.096], [664.871, 613.498], [733.223, 854.188], [808.602, 875.531], [891.73, 860.663], [983.405, 1196.09], [1084.5, 1369.16]]

    gammac11_gammasp18_z99 = [[19.6237, 0.111146], [21.6411, 0.223189], [23.866, 0.274115], [26.3195, 0.358174], [29.0253, 0.528627], [32.0092, 0.584111], [35.2999, 0.30166], [38.9289, 0.130602], [42.931, 0.110338], [47.3446, 0.200723], [52.2118, 0.387185], [57.5795, 0.487799], [63.4989, 0.390219], [70.027, 0.374343], [77.2261, 0.786262], [85.1653, 1.81275], [93.9208, 2.39607], [103.576, 1.26186], [114.224, 0.714934], [125.967, 0.900329], [138.917, 1.94702], [153.199, 3.80263], [168.948, 6.10194], [186.317, 5.93387], [205.472, 6.35122], [226.595, 5.86606], [249.89, 3.64342], [275.58, 2.00177], [303.911, 1.71701], [335.155, 3.852], [369.611, 11.809], [407.608, 32.2455], [449.513, 66.8202], [495.725, 76.1656], [546.688, 48.1534], [602.89, 22.2645], [664.871, 19.5835], [733.223, 25.7694], [808.602, 26.7158], [891.73, 26.807], [983.405, 35.2308], [1084.5, 39.7951]]

    ad_gammac1_z44 = [[12.3677, 4.5328*10**-8], [13.6392, 4.92245*10**-8], [15.0414, 6.53022*10**-8], [16.5877, 7.99851*10**-8], [18.293, 1.61246*10**-7], [20.1736, 2.72092*10**-7], [22.2476, 3.41284*10**-7], [24.5347, 2.35265*10**-7], [27.057, 1.06947*10**-7], [29.8386, 9.88366*10**-8], [32.9062, 1.08294*10**-7], [36.2891, 1.37662*10**-7], [40.0198, 1.4868*10**-7], [44.1341, 1.33551*10**-7], [48.6713, 1.62587*10**-7], [53.6749, 2.1685*10**-7], [59.193, 4.05485*10**-7], [65.2783, 5.27605*10**-7], [71.9893, 4.13017*10**-7], [79.3901, 4.36405*10**-7], [87.5519, 4.70944*10**-7], [96.5526, 7.01252*10**-7], [106.479, 1.29286*10**-6], [117.425, 1.39209*10**-6], [129.497, 2.31513*10**-6], [142.81, 3.13988*10**-6], [157.492, 2.8651*10**-6], [173.683, 2.759*10**-6], [191.538, 2.32976*10**-6], [211.229, 1.76462*10**-6], [232.945, 2.32342*10**-6], [256.893, 4.51366*10**-6], [283.303, 0.0000104844], [312.428, 0.000014779], [344.547, 0.0000158545], [379.968, 0.0000218529], [419.031, 0.0000217036], [462.109, 0.000023289], [509.616, 0.0000219583], [562.007, 0.0000193568], [619.785, 0.0000201716], [683.502, 0.0000216895]]

    gammac1_gammasp18_z44 = [[12.3677, 0.0311172], [13.6392, 0.0334056], [15.0414, 0.0487918], [16.5877, 0.0627363], [18.293, 0.165286], [20.1736, 0.32949], [22.2476, 0.432963], [24.5347, 0.252774], [27.057, 0.0776391], [29.8386, 0.0665465], [32.9062, 0.0729151], [36.2891, 0.0994583], [40.0198, 0.10683], [44.1341, 0.087744], [48.6713, 0.112212], [53.6749, 0.164133], [59.193, 0.39173], [65.2783, 0.548714], [71.9893, 0.372005], [79.3901, 0.387081], [87.5519, 0.415329], [96.5526, 0.708119], [106.479, 1.59336], [117.425, 1.70699], [129.497, 3.22239], [142.81, 4.6039], [157.492, 4.02434], [173.683, 3.74601], [191.538, 2.92427], [211.229, 1.94054], [232.945, 2.73984], [256.893, 6.35318], [283.303, 16.9224], [312.428, 24.401], [344.547, 26.1407], [379.968, 36.463], [419.031, 36.0703], [462.109, 38.6656], [509.616, 36.1054], [562.007, 31.1699], [619.785, 32.2879], [683.502, 34.6359]]

    gammac11_gammasp18_z44 = [[12.3677, 0.00140418], [13.6392, 0.00151401], [15.0414, 0.00205377], [16.5877, 0.002534], [18.293, 0.00539401], [20.1736, 0.00941955], [22.2476, 0.0119162], [24.5347, 0.00795346], [27.057, 0.00332454], [29.8386, 0.00303301], [32.9062, 0.00331839], [36.2891, 0.00427526], [40.0198, 0.00461245], [44.1341, 0.00407432], [48.6713, 0.00500496], [53.6749, 0.00679773], [59.193, 0.0133912], [65.2783, 0.0177044], [71.9893, 0.0134347], [79.3901, 0.014147], [87.5519, 0.0152479], [96.5526, 0.0233825], [106.479, 0.0449058], [117.425, 0.0483063], [129.497, 0.0819566], [142.81, 0.111654], [157.492, 0.101529], [173.683, 0.097363], [191.538, 0.0811893], [211.229, 0.0599185], [232.945, 0.0800218], [256.893, 0.159983], [283.303, 0.369216], [312.428, 0.512907], [344.547, 0.550989], [379.968, 0.745669], [419.031, 0.7468], [462.109, 0.802655], [509.616, 0.764985], [562.007, 0.682281], [619.785, 0.712444], [683.502, 0.766588]]

    z99_lims = np.zeros((len(gammac11_gammasp18_z99), 4))
    z44_lims = np.zeros((len(gammac11_gammasp18_z44), 4))

    for i in range(len(gammac11_gammasp18_z99)):
        z99_lims[i,0] = ad_gammac1_z99[i][0]
        z99_lims[i,1] = ad_gammac1_z99[i][1]
        z99_lims[i,2] = gammac1_gammasp18_z99[i][1]
        z99_lims[i,3] = gammac11_gammasp18_z99[i][1]

    for i in range(len(gammac11_gammasp18_z44)):
        z44_lims[i,0] = ad_gammac1_z44[i][0]
        z44_lims[i,1] = ad_gammac1_z44[i][1]
        z44_lims[i,2] = gammac1_gammasp18_z44[i][1]
        z44_lims[i,3] = gammac11_gammasp18_z44[i][1]

    fig = plt.figure(figsize=[14,7])
    ax = fig.add_subplot(121)
    plt.plot(z99_lims[:,0][:7], z99_lims[:,1][:7], linewidth=2.0, color='green', label='Adiabatic Spike, $\gamma_c = 1.0$')
    plt.plot(z99_lims[:,0][10:], z99_lims[:,1][10:], linewidth=2.0, color='green')
    plt.plot(z99_lims[:,0], z99_lims[:,2], linewidth=2.0,label='$\gamma_{sp}=1.8$, $\gamma_c = 1.0$')
    plt.plot(z99_lims[:,0], z99_lims[:,3], linewidth=2.0, label='$\gamma_{sp}=1.8$, $\gamma_c = 1.1$')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('$\\frac{<\sigma v>}{<\sigma v>_{therm}}$')

    plt.axhline(1.0, linestyle='--', color='black', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=2)
    plt.title('$\zeta = 0.9999$')
    plt.ylim([10**-6, 10**5])
    plt.xlim([10**1, 1.2*10**3])
    ax2 = fig.add_subplot(122)

    plt.plot(z44_lims[:,0], z44_lims[:,1], linewidth=2.0, color='green', label='Adiabatic Spike, $\gamma_c = 1.0$')
    plt.plot(z44_lims[:,0], z44_lims[:,2], linewidth=2.0,label='$\gamma_{sp}=1.8$, $\gamma_c = 1.0$')
    plt.plot(z44_lims[:,0], z44_lims[:,3], linewidth=2.0, label='$\gamma_{sp}=1.8$, $\gamma_c = 1.1$')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('$\\frac{<\sigma v>}{<\sigma v>_{therm}}$')

    plt.axhline(1.0, linestyle='--', color='black', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('$\zeta = 0.44$')
    plt.ylim([10**-8, 10**3])
    plt.xlim([10**1, 1.2*10**3])

    plt.legend(loc=2)

    plt.show()

    raw_input('wait for key')


def main():
    #[z99_ul, z99_corr] = likelihood_upper_limit3(0.9999, sourcemap, poisson=True)
    #file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/z99.pk1','wb')
    #pickle.dump([z99_ul, z99_corr],file)
    #file.close()

    #sourcemap = '6gev_srcmap_03.fits'
    #[z44_ul, z44_corr] = likelihood_upper_limit3(0.44, sourcemap)
    #file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/z44.pk1','wb')
    #pickle.dump([z44_ul, z44_corr],file)
    #file.close()

    sourcemap = 'box_srcmap_artificial_box.fits'
    [z44_artificial_ul, z44_artificial_corr] = likelihood_upper_limit3(0.44, sourcemap, poisson=False)
    #file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/z44_poisson.pk1','wb')
    #pickle.dump([z44_artificial_ul, z44_artificial_corr],file)
    #file.close()
    #raw_input('wait for key')
    #plotter([z0, z44, z85, z99])
    input('wait for key')




    brazil_dict = consolidate_brazil_lines('brazil_wide_box.pk1')
    file = open('z44.pk1','rb')
    [z44_ul, z44_corr_blank] = pickle.load(file)
    file.close()
    #Did this twice because you want to use NewMinuit to set limits, and MINUIT for correlations
    file = open('z44_corr.pk1','rb')
    [z44_ul_blank, z44_corr] = pickle.load(file)
    file.close()
    #correlationPlot(z44_corr, z99_corr)
    #make_ul_plot(z44_ul[:,0],brazil_dict,'UL','brazil_wide_box.pdf')
    #make_ul_plot(z44_ul,brazil_dict,'SIG', 'significance_wide_box.pdf')
    print("z44 significances = " + str(z44_ul[:,2]))
    print("Most signficant result = " + str(sigma_given_p(pvalue_given_chi2(-2.0*min(z44_ul[:,2]),1)))+ " sigma")
    print("at position " + str(energies[np.argmin(z44_ul[:,2])]) + " MeV")

    brazil_dict = consolidate_brazil_lines('brazil_narrow_box.pk1')
    file = open('z99.pk1','rb')
    [z99_ul, z99_corr_blank] = pickle.load(file)
    file.close()
    file = open('z99_corr.pk1','rb')
    [z99_ul_blank, z99_corr] = pickle.load(file)
    file.close()
    print("z99 significances = " + str(z99_ul[:,2]))
    print("Most signficant result = " + str(sigma_given_p(pvalue_given_chi2(-2.0*min(z99_ul[:,2]),1)))+ " sigma")
    print("at position " + str(energies[np.argmin(z99_ul[:,2])]) + " MeV")
    #make_ul_plot(z99_ul[:,0],brazil_dict,'UL','brazil_narrow_box.pdf')
    #make_ul_plot(z99,brazil_dict,'SIG', 'significance_narrow_box.pdf')
    correlationPlot(z44_corr, z99_corr)
    file = open('z_artificial.pk1','rb')
    [z44_artificial_ul, z44_artificial_corr] = pickle.load(file)
    file.close()
    brazil_dict = consolidate_brazil_lines('brazil_artificial_box.pk1')
    print("Most signficant result = " + str(sigma_given_p(pvalue_given_chi2(-2.0*min(z44_artificial_ul[:,2]),1)))+ " sigma")
    print("at position " + str(energies[np.argmin(z44_artificial_ul[:,2])]) + " MeV")

    #make_ul_plot(z44_artificial_ul[:,0], brazil_dict,'UL','brazil_artificial_box.pdf')
    #make_ul_plot(z_artificial_box, brazil_dict,'SIG', 'significance_artificial_box.pdf')

if __name__ == '__main__':
    main()
