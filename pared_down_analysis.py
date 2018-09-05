print("Loading required packages...")
#Math packages, NumPy
import warnings
warnings.filterwarnings("ignore")
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

#A derived class from the Science Tools BinnedAnalysis class
#Adds various methods which make life easier
class ExtendedAnalysis(BinnedAnalysis):

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

    def calculateCorrelationMatrix(self):
        correlation = np.zeros((np.array(self.covariance).shape))
        for a in np.array(self.covariance).shape[0]:
            for b in np.array(self.covariance).shape[1]:
                correlation[a,b] = self.covariance[a][b]/np.sqrt(float(self.covariance[a][a])*float(self.covariance[b][b]))
        self.correlation = correlation

    def getCorrelationMatrix(self):
        return self.correlation

    def getSpectrum(self):
        spectrum = np.zeros((num_ebins-1))
        for source in self.sourceNames():
            spectrum += self._srcCnts(source)
        return spectrum

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

num_ebins = 51 #1 more than the number of bins due to the fencepost problem
energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
ebin_widths = np.diff(energies)

def likelihood_upper_limit3(zeta, sourcemap):
    scale_factor = 1e-15
    min_box_flux = 0.0
    max_box_flux = 10**-9/scale_factor

    crit_chi2 = 2.71 #For 95% confidence one-sided upper limit with 1 degree of freedom
    scan_resolution = 200

    #Arrays to store results in
    box_flux = np.zeros((num_ebins-1, scan_resolution))

    #Instantiate the analysis objects
    print("Loading data...")
    obs_complete = BinnedObs(srcMaps=sourcemap, expCube='GC_binned_ltcube.fits', binnedExpMap='GC_binned_expcube.fits', irfs='P8R2_SOURCE_V6')
    print("obs loaded")
    like = ExtendedAnalysis(obs_complete, 'xmlmodel.xml', optimizer='MINUIT')
    like.free_all_sources()
    like.edit_parameter('Box_Component', 'Normalization', 0.0)
    like.freeze(like.par_index('Box_Component','Normalization'))
    likeobj = pyLike.Minuit(like.logLike)
    loglike = like.fit(verbosity=0, optObject=likeobj, covar=False)
    like.freeze_all_sources()
    print('LogLike = ' + str(loglike))
    #Loop through upper edge of box
    for index in range(6,48):
        print("Evaluating box with upper edge " + str(energies[index]) + " MeV in bin " + str(index))
        #Update the box spectrum
        box_width, integrated_box_flux = update_box_spectrum(energies[index], zeta)

        #Allow the GC Center to float along with the box
        like.thaw(like.par_index('Disk Component','Prefactor'))
        print("Scanning likelihood...")
        #Scan the likelihood profile to find best-fit value and upper limit
        x_range, l_range = like.scan('Box_Component', 'Normalization', min_box_flux, max_box_flux, scan_resolution)

        #Results
        box_flux[index, :] = l_range #Flux upper limit
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
    print(str(i) + " MC events")
    return brazil_dict

def UL_from_TS(ts):
    crit_chi2 = 5.0
    ul = np.zeros((ts.shape[0]))
    for i in range(len(ts)):
        ul[i] = np.argmin(ts[i,:]-crit_chi2)
    return ul


def main():
    sourcemap = 'GC_binned_srcmap.fits'
    z99 = likelihood_upper_limit3(0.9999, sourcemap)
    print(z99[20,:])
    plt.plot(energies[:-1], UL_from_TS(z99))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()
