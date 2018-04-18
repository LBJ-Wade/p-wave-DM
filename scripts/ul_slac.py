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
    box_maximum = 1e-8
    scale_factor = 1e-15
    
    box_string = []
    box_string.append(' <source name="Box Component" type="PointSource">\n')
    box_string.append('   <spectrum file="box_spectrum.dat" type="FileFunction">\n')
    box_string.append('     <parameter free="0" max="'+str(box_maximum*scale_factor**-1/box_width)+'" min="'+str(box_minimum*scale_factor**-1/box_width)+'" name="Normalization" scale="'+str(scale_factor)+'" value="'+str(box_flux*scale_factor**-1/box_width)+'"/>\n')
    box_string.append('   </spectrum>\n')
    box_string.append('  <spatialModel type="SkyDirFunction">\n')
    box_string.append('     <parameter free="0" max="360" min="-360" name="RA" scale="1" value="266.417" />\n')
    box_string.append('     <parameter free="0" max="90" min="-90" name="DEC" scale="1" value="-29.0079" />\n')
    box_string.append('   </spatialModel>\n')
    box_string.append(' </source>\n')

    pedestal_maximum = 1e-8
    pedestal_minimum = 0
    
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

#Likelihood from gtlikelihood module

#Likelihood from gtlikelihood module
def loglikelihood3(energy, box_flux, obs, z, get_like=False, covar=False):

    #Next, edit the XML file to have the right flux and spectrum
    edit_box_xml(energy, box_flux, z)

    #Do the fitting
    like1 = BinnedAnalysis(obs, 'xmlmodel_fixed_box.xml', optimizer='DRMNFB')
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
        q = like1.fit(verbosity=0, tol=1e10, optObject=like1obj, covar=covar, optimizer='DRMNFB')
        
        if get_like:
            return q, like1
        else:
            return q
        
def likelihood_upper_limit3(z):
    
    #Array to hold results of flux upper limit calculation
    num_ebins = 51 #1 more than the number of bins due to the fencepost problem
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
    ebin_widths = np.diff(energies)
    
    sourcemap = '6gev_srcmap_03_pedestal.fits'#'box_srcmap_artificial_box.fits'#'6gev_srcmap_complete.fits'
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
            srcmap_complete[9].data = srcmap_complete[9].data[:window_low+1]
            m = srcmap_complete[9]

            os.system('rm srcmap_low.fits')
            b.header['DSVAL4'] = str()+':'+str()
            hdulist = pyfits.HDUList([a, srcmap_complete[1], b, c, d, e, f, g, h, m])
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
            srcmap_complete[9].data = srcmap_complete[9].data[window_high:]
            m = srcmap_complete[9]

            os.system('rm srcmap_high.fits')
            hdulist = pyfits.HDUList([a, srcmap_complete[1], b, c, d, e, f, g, h, m])
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
        
        calculation = 'poisson'
        obs_complete = BinnedObs(srcMaps=sourcemap, expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='CALDB')
        edit_box_xml(100000.0,1e-15,0.0)
        like = BinnedAnalysis(obs_complete, 'xmlmodel_fixed_box.xml', optimizer='MINUIT')
        like.tol=1e-8
        like_obj = pyLike.Minuit(like.logLike)
        like.fit(verbosity=3,optObject=like_obj)
        like.writeXml('xmlmodel_fixed_box.xml')
        
        #Flucuate the window data
        f = pyfits.open(sourcemap)
        poisson_data = np.zeros((len(range(max(window_low,0), min(window_high, 49))), 50, 50))
        q = 0
        for bin in range(max(window_low,0), min(window_high,49)):
            for source in like.sourceNames():
                for j in range(3,10):
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
            
        if calculation == 'complete':
             obs_calculation=obs_complete
        else:
            obs_calculation= obs_poisson
        
            
        print "Finding Upper Limit..."
        null_likelihood = loglikelihood3(energies[index], 1.0e-15, obs_calculation, z)
        delta_loglike = 0.0
        
        
        #increase box flux until likelihood > 2sigma over null likelihood
        box_flux[index] = 3.e-15
        crit_chi2 = 2.706
        while delta_loglike <crit_chi2:
            print "delta loglike = " + str(delta_loglike)
            print "flux = " + str(box_flux[index]) + " likelihood = " + str(loglikelihood3(energies[index], box_flux[index], obs_calculation, z))
            box_flux[index]*=3.0                
            delta_loglike = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
            
        box_flux[index]*=1.0/3.0
        delta_loglike = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
        print "Delta loglike = " + str(delta_loglike)
        while delta_loglike <crit_chi2:
            print "delta loglike = " + str(delta_loglike)
            print "flux = " + str(box_flux[index]) + " likelihood = " + str(loglikelihood3(energies[index], box_flux[index], obs_calculation, z))
            box_flux[index]*=1.5
            delta_loglike = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
            
        box_flux[index]*=1.0/1.5
        delta_loglike = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
        
        while delta_loglike <crit_chi2:
            print "delta loglike = " + str(delta_loglike)
            print "flux = " + str(box_flux[index]) + " likelihood = " + str(loglikelihood3(energies[index], box_flux[index], obs_calculation, z))
            box_flux[index]*=1.03
            delta_loglike = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
            
        box_flux[index]*= 1.0/1.03
        delta_loglike = 2.0*(loglikelihood3(energies[index], box_flux[index], obs_calculation,z)-null_likelihood)
        
        print "delta log like = " + str(delta_loglike)
        
    
        
        calc_cov = False
        if calc_cov:
            like1 = BinnedAnalysis(obs_calculation, 'xmlmodel_fixed_box.xml', optimizer='DRMNFB')
            
            like1.thaw(like1.par_index('Disk Component','Index'))
            like1.thaw(like1.par_index('Disk Component','Prefactor'))
            like1.thaw(like1.par_index('Box Component','Normalization'))
            like1.tol=1e-5
            like1obj = pyLike.Minuit(like1.logLike)
            like1.fit(verbosity=0,optObject=like1obj, covar=False)

            like1.writeXml('xmlmodel_fixed_box.xml')
            
            like2 = BinnedAnalysis(obs_calculation, 'xmlmodel_fixed_box.xml', optimizer='NewMinuit')
            like2.tol=1e-8
            like2obj = pyLike.Minuit(like1.logLike)
            like2.fit(verbosity=3,optObject=like1obj, covar=True)
            
            
            #ul = UpperLimit(like1,'Box Component')
            #ul.compute(emin=100.0,emax=500000, delta=3.91)
            
            #box_flux_bayesian[index] = float(ul.bayesianUL()[0])
            #box_flux_frequentist[index] = float(ul.results[0].value)            
            print like2.covariance
            print 'Return code: ' + str(like2obj.getRetCode())
            cov = like2.covariance
            corr[index] = cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])
            corr2[index] = cov[0][2]/np.sqrt(cov[0][0]*cov[2][2])
            print "Correlations:"
            print corr[index]
            print corr2[index]
            #if like2obj.getRetCode()!=0:
            plot_spectrum(like2, energies, index, window_low, window_high)
    
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

mc_ul = likelihood_upper_limit3(z=0.44)
file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/wide_box/'+os.environ['LSB_JOBID']+'.pk1','wb')
pickle.dump(mc_ul,file)
file.close()
