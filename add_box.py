#This code adds photons with a box-like spectrum to a point source located at the galactic center
#The output is a new counts cube. Goal here: validate the upper limit code
#The new counts cube can be added to gtsrcmaps and re-run the whole pipeline, we should see a nice bump over the uncertainty

import numpy as np
import matplotlib.pyplot as plt
import pyfits

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

def make_random(x,g):
    cdf = np.cumsum(g)/np.sum(g)
    return x[np.argmin(np.abs(cdf-np.random.rand(1)[0]))]

#looking at TS distribution
"""
import pickle
import scipy.stats as stats

file = open('ts_hist.pk1', 'rb')
g = pickle.load(file)
file.close()

x = np.zeros((len(g), len(g[0])))
i = 0
for entry in g:
    x[i,:] = -1.0*entry
    i += 1

bins = 10**np.linspace(-2, 1.03, 50)
plt.hist(x[np.nonzero(x)], bins, color='blue', alpha=0.4, label='Monte Carlo Data')
plt.plot(bins, 700*stats.chi2.pdf(bins,df=3),color='red', linewidth=2.0, label='Chi^2, 3 Degrees of Freedom')
plt.plot(bins, 200*stats.chi2.pdf(bins, df=1),color='green', linewidth=2.0, label='Chi^2, 1 Degree of Freedom')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('TS Value')
plt.ylabel('Counts')
plt.xlim([np.min(bins), np.max(bins)])
plt.legend(loc=3)
plt.savefig('plots/ts_hist.pdf', bbox_inches='tight')
plt.show()
"""

num_ebins = 51
energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
ebin_widths = np.diff(energies)
box_flux = 2.0*10**-10#total flux
endpoint_e = 100000#MeV, aka 100 GeV
dnde = box_flux/endpoint_e #flux per MeV

box_counts = np.zeros((num_ebins-1))
exposures = np.zeros((num_ebins-1))

#Time to figure out the ideal window size!
#To the right: we know the energy resolution- we just need enough bins to make sure 95% of the signal is captured
#To the left: we want the counts fraction to be less than 5%?
#Given a slope of -2 in counts:
#N(i) ~ i^-2 where i is bin number
#Also, bin size increases as you go right - by a factor of 1.103
#so for the bin immediately to the left, you have 1.103X fewer counts from the box, and

exposurefile = pyfits.open('6gev_exposure.fits')
for i in range(len(exposurefile[0].data)-1):
    exposures[i] = exposurefile[0].data[i][25, 25]
exposurefile.close()



#Account for energy dispersion
x_fine_grid = np.linspace(0.0, 800000, 10000)
pure_box = np.concatenate([1.0+np.zeros((int(np.argmin(np.abs(x_fine_grid-endpoint_e))))),np.zeros((10000-int(np.argmin(np.abs(x_fine_grid-endpoint_e)))))])

#Sigma here is the absolute energy resolution as a function of energy
sigma = e_res(endpoint_e)*endpoint_e*10000./800000.0

dispersion = blur(np.linspace(0,6*sigma,6*sigma),3*sigma,sigma)
convolved_pure_box = np.convolve(pure_box, dispersion,'same')

convolution = np.zeros((num_ebins-1))
for i in range(len(convolution)):
    convolution[i] = convolved_pure_box[np.argmin(np.abs(x_fine_grid-energies[i]))]

"""
plt.yscale('log')
plt.xscale('log')
plt.plot(energies[:-1], dnde*ebin_widths*exposures*convolution, color='blue',linewidth=2.0, label='Convolved Box')
plt.plot(energies[:-1], dnde*ebin_widths*exposures, color='red',linewidth=2.0, label='Flat Spectrum')
plt.axvline(endpoint_e, color='black', linestyle='--', linewidth=0.5,label='Box Edge')
plt.legend(loc=4)
plt.xlim([10000, 800000])
plt.xlabel('Energy [MeV]')
plt.ylabel('Counts')
plt.savefig('plots/Box_convolution.png',bbox_inches='tight')
plt.show()
"""

print "OG counts: " + str(np.sum(dnde*ebin_widths*exposures))
box_counts = dnde*ebin_widths*exposures*convolution
print "post-convolution counts: " + str(np.sum(box_counts))
print "Total flux: " + str(np.sum(box_counts)/(4.5*10**11))
the_index = np.argmax(box_counts)
print box_counts
print "Other way of calculating: " + str(box_counts[the_index]*energies[the_index]/ebin_widths[the_index]/exposures[the_index])


f = pyfits.open('6gev_srcmap_03.fits')
q = 0


for i in range(num_ebins-1):
    model_counts = np.zeros((1,len(f[6].data[i].ravel())))[0]
    num_photons = np.random.poisson(box_counts[i])
    print "num photons = " + str(num_photons) + " or " + str(100.*num_photons/np.sum(np.sum(f[0].data[i]))) + "% of total counts in bin " + str(i)
    print "sqrt(num photons) = " + str(np.sqrt(num_photons))
    print "flux ul 95% = " + str(2*np.sqrt(num_photons)/num_photons)
    for photon in range(num_photons):
        phot_loc = int(make_random(np.arange(0,len(model_counts), 1), f[6].data[i].ravel()))
        model_counts[phot_loc] += 1
        

    model_counts = model_counts.reshape(50, 50)
    f[0].data[i] += model_counts
f.writeto('box_srcmap_artificial_box.fits')
f.close()
