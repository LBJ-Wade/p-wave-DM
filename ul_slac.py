print "initializing..."
import math
import numpy as np
from scipy.special import gammainc
import pickle
import pylab as plt
from matplotlib import cm, colors
import pyfits
#from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
#from astropy import wcs
import sys
import os
from BinnedAnalysis import *
from SummedLikelihood import *
import pyLikelihood as pyLike
from gt_apps import evtbin, srcMaps, gtexpcube2

print "Done!"


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
#Just a gaussian function, representing the energy dispersion of the detector
def blur(x,offset,sigma):
    return np.exp(-1.0*(x-offset)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#We need to account for exposure. How to do this?
#We have counts in each bin, model for counts in each bin
#Also have model of box- in flux
#Need to multiply box by value of exposure in each bin, to get expected number of counts

#What's the exposure as a function of energy?
#exposure_file = pyfits.open('/scratch/johnsarc/'+os.environ['LSB_JOBID']+'/8gev_exposure.fits')
#exp_energies = np.zeros((len(exposure_file[1].data)))
#for i in range(len(exp_energies)):
#    exp_energies[i] = float(exposure_file[1].data[i][0])
def make_random(x,g):
    cdf = np.cumsum(g)/np.sum(g)
    return x[np.argmin(np.abs(cdf-np.random.rand(1)[0]))]



def edit_box_xml(energy, box_flux):
    #Choose between a wide and narrow box
    box_beginning = energy-1000.0#energy-100 -> 100 MeV wide box
    
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
    print "Log Likelihood = " + str(q)
    return q

#Likelihood from gtlikelihood module
def likelihood_upper_limit3():
    #Array to hold results of flux upper limit calculation
    num_ebins = 51 #1 more than the number of bins due to the fencepost problem
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
    ebin_widths = np.diff(energies)
    
    sourcemap = '6gev_srcmap_03.fits'#'6gev_srcmap_complete.fits'
    box_flux = np.zeros((num_ebins-1))
    ts = np.zeros((num_ebins-1))
    gll_counts = np.zeros((num_ebins-1))
    
    #reconstructed_spectra = np.zeros((num_ebins-1, num_ebins-1))
    #Loop through upper edge of box
    for index in range(6,48):
        print "Calculating upper limit in bin " + str(index)
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
            obs_low = BinnedObs(srcMaps='srcmap_low.fits', expCube='6gev_ltcube.fits', binnedExpMap='exposure_low.fits', irfs='CALDB')
            like_low = BinnedAnalysis(obs_low, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
            summedLike.addComponent(like_low)


        if index<48:
            obs_high = BinnedObs(srcMaps='srcmap_high.fits', expCube='6gev_ltcube.fits', binnedExpMap='exposure_high.fits', irfs='CALDB')
            like_high = BinnedAnalysis(obs_high, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
            summedLike.addComponent(like_high)

        summedLike.ftol = 1e-8
        summedLike.fit(verbosity=0)
        summedLike.writeXml('xmlmodel_free.xml')
        for k in range(len(summedLike.params())):
            summedLike.freeze(k)
        summedLike.writeXml('xmlmodel_fixed.xml')
        
        
        obs_complete = BinnedObs(srcMaps=sourcemap, expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='CALDB')
        
        #Flucuate the window data
        like = BinnedAnalysis(obs_complete, 'xmlmodel_fixed.xml', optimizer='NEWMINUIT')
        like.tol=1e-8
        like_obj = pyLike.Minuit(like.logLike)
        like.fit(verbosity=3,optObject=like_obj)
        complete_spectrum = like.nobs#
        gll_counts[index] =float(like._srcCnts('gll_iem_v05')[index]/like._srcCnts('Disk Component')[index])
        
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
        
        obs_calculation = obs_poisson #obs_poisson or obs_complete
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
        box_flux[index] = 10.**-13

        null_likelihood = loglikelihood3(energies[index], 10.**-25, obs_calculation)
        old_likelihood=null_likelihood
        while loglikelihood3(energies[index], box_flux[index], obs_calculation)<old_likelihood:
            old_likelihood = loglikelihood3(energies[index], box_flux[index], obs_calculation)
            box_flux[index] *= 1.3
        box_flux[index] = box_flux[index]/1.3
        ts[index] = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood)

        #increase box flux until likelihood > 2sigma over null likelihood
        box_flux[index] = 10.**-13
        while sigma_given_p(pvalue_given_chi2(2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood),3))<2.0:
            box_flux[index]*=5.0
        box_flux[index]*=1.0/5.0
        while sigma_given_p(pvalue_given_chi2(2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood),3))<2.0:
            box_flux[index]*=1.1
        box_flux[index]*=1.0/1.1
        while sigma_given_p(pvalue_given_chi2(2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation)-null_likelihood),3))<2.0:
            box_flux[index]*=1.03
        
        print box_flux[index]
        """
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
        """
    return box_flux, ts#box_flux#, reconstructed_spectra


mc_ul, ts = likelihood_upper_limit3()
file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/'+os.environ['LSB_JOBID']+'.pk1','wb')
pickle.dump([mc_ul,ts],file)
file.close()
