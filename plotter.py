import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from scipy.special import gammainc
from scipy.misc import factorial
from scipy.signal import savgol_filter

import math
import pickle
#from BinnedAnalysis import *
import matplotlib.colors as colors
from matplotlib.pyplot import rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization.wcsaxes.frame import EllipticalFrame

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.signal import convolve2d

#from upper_limit import AnalyticAnalysis

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
    rcParams['axes.labelsize'] = 14
    rcParams['font.size'] = 14
    rcParams['axes.titlesize'] =16
    rcParams['legend.fontsize'] = 10
    rcParams['xtick.labelsize'] =12
    rcParams['ytick.labelsize'] =12
    rcParams['figure.figsize'] = fig_size
    rcParams['xtick.major.size'] = 8
    rcParams['ytick.major.size'] = 8
    rcParams['xtick.minor.size'] = 4
    rcParams['ytick.minor.size'] = 4
    rcParams['xtick.major.pad'] = 4
    rcParams['ytick.major.pad'] = 4
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['figure.subplot.left'] = 0.16
    rcParams['figure.subplot.right'] = 0.92
    rcParams['figure.subplot.top'] = 0.90
    rcParams['figure.subplot.bottom'] = 0.12
    rcParams['text.usetex'] = True
    rc('text.latex', preamble=r'\usepackage{amsmath}')
setup_plot_env()

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
def sigma_given_p(p):
    x = np.linspace(-200, 200, 50000)
    g = 1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.)
    c = np.cumsum(g)/sum(g)
    value = x[np.argmin(np.abs(c-(1.0-p)))]
    return value


def poisson(x,k):
    return x**k*math.exp(-1.0*x)/math.gamma(k+1)

def listToArray(dict):
    """
    A method to convert a dictionary's list entries to numpy arrays
    """
    for key in dict.keys():
        dict[key] = np.array(dict[key])
    return dict

#Given an expected number of counts and an observed number of counts, what's the significance?
#i.e. if we expect 10 counts and see 15, how many sigma deviation is that?
def frequentist_counts_significance(observed_counts, mean_counts):
    if observed_counts>mean_counts:
        the_sum = 0.0
        if np.abs(observed_counts-mean_counts)<1.0:
            pvalue = 1.0-poisson(mean_counts, observed_counts)
            sig = sigma_given_p(pvalue)
            return float(sig)
        else:
            for k in np.arange(max(2.0*mean_counts-observed_counts, 0), max(observed_counts, 0)):
                the_sum += poisson(k, mean_counts)
            pvalue = 1.0-the_sum
            sig = sigma_given_p(pvalue)
            return float(sig)
    elif observed_counts<mean_counts:
        if np.abs(observed_counts-mean_counts)<1.0:
            pvalue = 1.0 - poisson(mean_counts, observed_counts)
            sig = sigma_given_p(pvalue)
            return float(-1.0*sig)
        the_sum = 0.0
        for k in np.arange(max(observed_counts,0), 2.0*mean_counts-observed_counts):
            the_sum += poisson(k, mean_counts)
        pvalue = 1.0-the_sum
        sig = sigma_given_p(pvalue)
        return float(-1.0*sig)
    else:
        return 0

#Given a number of counts, what is the proper confidence interval
#Roughly follows sqrt(counts), but differs at low counts
#No background, just counting statistics. Ref: PDG 38.71a, b
#Can reproduce table PDG 38.3
def frequentist_upper_lower_limits(observed_counts, alpha):
    upper_limit = 0.5*chi_square_quantile(2.0*(observed_counts+1), 1.0-alpha)
    lower_limit = 0.5*chi_square_quantile(2.0*observed_counts, alpha)
    return lower_limit, upper_limit


def ra_dec_to_l_b(ra_input, dec_input):
    l = SkyCoord(ra=ra_input*u.degree,dec=dec_input*u.degree).galactic.l.degree
    b = SkyCoord(ra=ra_input*u.degree,dec=dec_input*u.degree).galactic.b.degree
    return l, b

def l_b_to_ra_dec(l_input, b_input):
    ra = SkyCoord(l=l_input*u.degree,b=b_input*u.degree,frame='galactic').icrs.ra.degree
    dec = SkyCoord(l=l_input*u.degree,b=b_input*u.degree,frame='galactic').icrs.dec.degree
    return ra, dec



#0.04 degrees: 15798.5084759
#0.03 degrees: 15797.2262217
#0.02 degrees: 15803.1695199
#0.01 degrees: 15808.9181639
#pt source: 15808.1352914
#code to make a python list of dictionaries of 3fgl sources


def spectralPlot():
    file = open('plotsData/goodBoxFit.npy','rb')
    g = np.load(file)
    file.close()

    fig = plt.figure(figsize=[7,7])
    ax = fig.add_subplot(111)
    num_ebins = 51
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)

    plt.plot(energies[:-1], g[0], linewidth=2., color='blue', label='Total Reconstructed Spectrum')
    plt.errorbar(energies[:-1], g[1], xerr=0, yerr=np.sqrt(g[1]), fmt='o', color='black',label='Data + Injected Signal')
    plt.plot(energies[:-1], g[2], linewidth=2.0, color='red', ls='-.', label='Reconstructed Signal')
    rcParams['legend.fontsize'] = 16

    plt.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([8000.0, 800000])
    ax.set_ylim([5*10**-1, 10**3])
    #plt.title(filename)
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Counts in the ROI')
    #plt.show()
    plt.savefig('plots/good_box_fit.pdf', bbox_inches='tight')

def correlationPlot():
    fig = plt.figure(figsize=[7,7])
    ax = fig.add_subplot(111)

    file = open('plotsData/wideBoxCorrelations.npy','rb')
    correlation1 = np.load(file, encoding='latin1')
    file.close()

    file = open('plotsData/narrowBoxCorrelations.npy','rb')
    correlation2 = np.load(file, encoding='latin1')
    file.close()
    num_ebins = 51
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)

    plt.plot(energies[6:48], correlation1[:,0][6:48], linewidth=2.0, color='blue', label='$\zeta=0.44$')
    plt.plot(energies[6:48], correlation2[:,0][6:48], linewidth=2.0, color='red', label='$\zeta=0.9999$')
    plt.axhline(0.0, color='black', linestyle='--', linewidth=0.5)
    plt.xscale('log')
    plt.ylim([-1.0, 0.1])
    plt.xlim([energies[6], energies[47]])
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Box Upper Edge [MeV]')
    rcParams['legend.fontsize'] = 16

    plt.legend()
    plt.grid(True)
    plt.savefig('plots/correlation_coefficients.pdf',bbox_inches='tight')
    #plt.show()

def residmapComparison():
    """
    Making Figure 2 in the paper (comparing the residuals between the GC point source model and GC extended source model)
    """
    srcmap001 = fits.open('dataFiles/6gev_srcmap_001.fits')
    srcmap03 = fits.open('dataFiles/6gev_srcmap_03.fits')

    image_data = fits.getdata('dataFiles/6gev_image.fits')
    filename = get_pkg_data_filename('dataFiles/6gev_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    #Given the results of the fit, calculate the model
    modelData001 = np.zeros(srcmap001[0].shape)
    modelData03 = np.zeros(srcmap03[0].shape)

    file = open('plotsData/fitResults001.pk1','rb')
    fit001 = pickle.load(file)
    fit001 = listToArray(fit001)
    file.close()

    file = open('plotsData/fitResults03.pk1','rb')
    fit03 = pickle.load(file)
    fit03 = listToArray(fit03)
    file.close()


    for source in fit001:
        the_index = srcmap001.index_of(source)

        modelData001 += fit001[source][:, None, None]*srcmap001[the_index].data[:-1, :, :]/np.sum(np.sum(srcmap001[the_index].data, axis=2), axis=1)[:-1, None, None]
    for source in fit03:
        the_index = srcmap03.index_of(source)
        modelData03 += fit03[source][:, None, None]*srcmap03[the_index].data[:-1, :, :]/np.sum(np.sum(srcmap03[the_index].data, axis=2), axis=1)[:-1, None, None]

    fig = plt.figure(figsize=[12, 4.5])

    vmin = -25.0
    vmax = 25.0
    cbStep = 5.0
    ax = fig.add_subplot(121, projection=wcs)
    ax=plt.gca()
    ax.tick_params(direction='in')
    c = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)
    mappable=plt.imshow((image_data-np.sum(modelData001,axis=0)),cmap='seismic',origin='lower',vmin=vmin, vmax=vmax, interpolation='gaussian')#
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.title('GC Point Source ($>6$ GeV)')
    cb = plt.colorbar(mappable, label='Residual counts per pixel', pad=0.01,ticks=np.arange(vmin, vmax+cbStep, cbStep))
    cb.ax.tick_params(width=0)


    ax2=fig.add_subplot(122, projection=wcs)
    ax2 = plt.gca()
    c2 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax2.get_transform('galactic'))
    ax2.add_patch(c2)
    mappable2 = plt.imshow((image_data-np.sum(modelData03,axis=0)), cmap='seismic',origin='lower',vmin=vmin, vmax=vmax, interpolation='gaussian')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.title('GC Extended Source ($>6$ GeV)')
    cb2 = plt.colorbar(mappable2, label='Residual counts per pixel', pad=0.01, ticks=np.arange(vmin, vmax+cbStep, cbStep))
    cb2.ax.tick_params(width=0)
    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.13, left=0.04, bottom=0.13, top=0.92)
    plt.savefig('plots/residComparison.pdf',bbox_inches='tight')
    #plt.show()

def dataModel():
    """
    Making Figure 1 in the paper (Data versus model)
    """
    srcmap001 = fits.open('dataFiles/6gev_srcmap_001.fits')
    srcmap03 = fits.open('dataFiles/6gev_srcmap_03.fits')

    image_data = fits.getdata('dataFiles/6gev_image.fits')
    filename = get_pkg_data_filename('dataFiles/6gev_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    #Given the results of the fit, calculate the model
    modelData001 = np.zeros(srcmap001[0].shape)
    modelData03 = np.zeros(srcmap03[0].shape)

    file = open('plotsData/fitResults001.pk1','rb')
    fit001 = pickle.load(file)
    fit001 = listToArray(fit001)
    file.close()

    file = open('plotsData/fitResults03.pk1','rb')
    fit03 = pickle.load(file)
    fit03 = listToArray(fit03)
    file.close()


    for source in fit001:
        the_index = srcmap001.index_of(source)

        modelData001 += fit001[source][:, None, None]*srcmap001[the_index].data[:-1, :, :]/np.sum(np.sum(srcmap001[the_index].data, axis=2), axis=1)[:-1, None, None]
    for source in fit03:
        the_index = srcmap03.index_of(source)
        modelData03 += fit03[source][:, None, None]*srcmap03[the_index].data[:-1, :, :]/np.sum(np.sum(srcmap03[the_index].data, axis=2), axis=1)[:-1, None, None]

    fig = plt.figure(figsize=[12, 4.5])

    vmin = 0
    vmax = 70.0
    cbStep = 10.0
    ax = fig.add_subplot(121, projection=wcs)
    ax=plt.gca()
    ax.tick_params(direction='in')
    c = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)
    mappable=plt.imshow((image_data),cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=vmin, vmax=vmax, interpolation='gaussian')#
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.title('Data ($>6$ GeV)')
    cb = plt.colorbar(mappable, label='Counts per pixel', pad=0.01,ticks=np.arange(vmin, vmax+cbStep, cbStep))
    cb.ax.tick_params(width=0)


    ax2=fig.add_subplot(122, projection=wcs)
    ax2 = plt.gca()

    sources = []
    sources.append({
    'Name':'3FGL J1745.3-2903c',
    'RA':266.3434922,
    'DEC':-29.06274323,
    'color':'xkcd:bright light blue'})

    sources.append({
    'Name':'1FIG J1748.2-2816',
    'RA':267.1000722,
    'DEC':-28.27707114,
    'color':'xkcd:fire engine red'
    })

    sources.append({
    'Name':'1FIG J1746.4-2843',
    'RA':266.5942898,
    'DEC':-28.86244442,
    'color':'xkcd:fluorescent green'
    })

    sources.append({
    'Name':'Galactic Center',
    'RA':266.417,
    'DEC':-29.0079,
    'color':'black'
    })

    #Add source names:
    for source in sources:
        l, b = ra_dec_to_l_b(source['RA'], source['DEC'])
        ax2.scatter(l, b, color=source['color'],marker='x',s=45.0, transform=ax2.get_transform('galactic'), label=source['Name'])

    c2 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax2.get_transform('galactic'))
    ax2.add_patch(c2)
    mappable2 = plt.imshow((np.sum(modelData03,axis=0)), cmap='inferno',norm=colors.PowerNorm(gamma=0.6),origin='lower',vmin=vmin, vmax=vmax, interpolation='gaussian')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.title('Model ($>6$ GeV)')
    cb2 = plt.colorbar(mappable2, label='Counts per pixel', pad=0.01, ticks=np.arange(vmin, vmax+cbStep, cbStep))
    cb2.ax.tick_params(width=0)
    rcParams['legend.fontsize'] = 10

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.5)
    leg.get_frame().set_edgecolor('white')
    text1 = leg.get_texts()
    for text in text1:
        text.set_color('black')

    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.13, left=0.04, bottom=0.13, top=0.92)
    #plt.show()
    plt.savefig('plots/dataModelComparison.pdf',bbox_inches='tight')

def tsDistribution():
    file = open('plotsData/savedMC_TS.npy', 'rb')
    g = np.load(file)
    file.close()
    bins = 10**np.linspace(-2.0, 1.2, 50)
    h = np.histogram(-2.0*np.concatenate(g), bins=bins)
    fig = plt.figure(figsize=[7,7])
    plt.fill_between(bins[:-1], 0.0, h[0]/np.diff(h[1]), step='pre', alpha=0.4, color='black', label='Monte Carlo Data')
    plt.plot(bins, 3500.*chi_square_pdf(1.0, bins), label='$\chi^2$, 1 d.o.f.', linewidth=3.0)
    plt.plot(bins, 3500.*chi_square_pdf(2.0, bins), label='$\chi^2$, 2 d.o.f.', linewidth=3.0)

    #plt.plot(bins, 1000.*chi_square_pdf(1.0, bins)+1000.*chi_square_pdf(2.0, bins), label='k=1.5', linewidth=2.0)
    plt.ylim([10**0, 10**4])
    plt.xlim([10**-2, 10**1.2])
    plt.grid('on', linestyle='--', linewidth=0.5, color='black')
    plt.ylabel('Counts')
    plt.xlabel('TS Value [$-2\Delta$log$\mathcal{L}$]')
    rcParams['legend.fontsize'] = 16
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('plots/ts_hist.pdf',bbox_inches='tight')
    #plt.show()

def brazilPlot(ulFile, brazilFile, plt_title):
    num_ebins = 51
    energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)

    file = open(brazilFile,'rb')
    brazilData = np.load(file)
    file.close()
    mcLimits = np.zeros((len(brazilData),num_ebins-1))
    i = 0
    for entry in brazilData:
        mcLimits[i,:] = entry
        i += 1

    file = open(ulFile, 'rb')
    dataLimits = np.load(file)[:,0]
    file.close()

    trials = len(mcLimits)
    lower_95 = np.zeros((num_ebins-1))
    lower_68 = np.zeros((num_ebins-1))
    upper_95 = np.zeros((num_ebins-1))
    upper_68 = np.zeros((num_ebins-1))
    median = np.zeros((num_ebins-1))
    for i in range(num_ebins-1):
        lims = mcLimits[:,i]
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


    fig = plt.figure(figsize=[7,7])
    ax = fig.add_subplot(111)
    #Plotting uppper limit
    #ax.plot(energies[:-1],median,color='black',linewidth=1,linestyle='--', label='Median MC')
    ax.fill_between(energies[:-1][6:48], lower_95, upper_95, color='yellow', label='95\% Containment')
    ax.fill_between(energies[:-1][6:48], lower_68, upper_68, color='#63ff00',label='68\% Containment')
    ax.plot(energies[:-1][6:48],dataLimits[6:48], marker='.', markersize=13.0,color='black',linewidth=2, label='95\% Confidence Upper Limit')
    ax.plot(energies[:-1][6:48], median, linestyle='--', linewidth=0.5, color='black', label='Median Expected')
    rcParams['legend.fontsize'] = 16

    #Uncomment the following line to show a dot where an injected signal lives
    ax.errorbar(np.array([1e5]), np.array([3.0*10**-10]), xerr=0, yerr=0.0, color='blue', markersize=10, fmt='o', label='Injected Signal')

    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel('Flux Upper Limit [ph s$^{-1}$ cm$^{-2}$]')
    plt.xlabel('Energy [MeV]')
    plt.legend(loc=3)

    ax.set_xlim([energies[6], energies[47]])
    ax.set_ylim([2*10**-12, 2*10**-9])
    plt.savefig('plots/'+str(plt_title),bbox_inches='tight')
    #plt.show()

def theoryPlotter():
    ad_gammac1_z99 = [[19.62371415969091, 1.7519089067965915*10**-6], [21.641132498163635,
       7.505354915564719*10**-6], [23.865951776090906,
       3.382227677967238*10**-6], [26.319493872472727,
       0.000011864830943523662], [29.025272664672727,
       0.000017120772864963094], [32.00921937712727,
       0.00001888879794471948], [35.29993109694545,
       0.000010167395180778949], [38.92894483829091,
       2.3007390583952284*10**-6], [42.931039781927275,
       2.0645052779880693*10**-6], [47.34457058660001,
       3.293618862496714*10**-6], [52.21183496636364,
       5.320547623811047*10**-6], [57.57947905705454,
       6.367015251029276*10**-6], [63.498944456109086,
       5.644323802389934*10**-6], [70.02696122076362,
       5.629507396437339*10**-6], [77.22609154872727,
       9.433385358967202*10**-6], [85.16532935207273,
       0.00005704693476946674], [93.92076147050909,
       0.00007408808847413974], [103.5762968606,
       0.00004147669164374839], [114.22447074945455,
       9.964304433669993*10**-6], [125.96733145956364,
       0.000030330673734611892], [138.91741840369093,
       0.00006308395932769567], [153.19884062274545,
       0.00011704974690483896], [168.94846620278182,
       0.00018165428425927183], [186.31723397021815,
       0.00017826375614803374], [205.47160003598182,
       0.00019103568854747003], [226.59513305192723,
       0.00017892273373828482], [249.89027346759997,
       0.000116762017752119], [275.58027364783635,
       0.000025250284240196542], [303.9113374441091,
       0.000023521974783895208], [335.15497972505455,
       0.00012540766446969927], [369.6106284786545,
       0.0003515480401783166], [407.60849442385455,
       0.0009102039141231921], [449.5127356330182,
       0.0018883719332044608], [495.7249474937636,
       0.0021564506758517503], [546.6880114567272,
       0.0013547580818977097], [602.8903394544,
       0.000651262624195416], [664.8705546677091,
       0.0005819009590256931], [733.2226534998182,
       0.0007543316791189843], [808.6016982268181,
       0.0007843458021098378], [891.7300948823455,
       0.0007912976582441107], [983.4045165408909,
       0.0010250605430997452], [1084.5035383499455,
       0.0011540784699198788]]

    gammac1_gammasp18_z99 = [[19.62371415969091, 1.6741543737182027], [21.641132498163635,
       4.10127800034358], [23.865951776090906,
       5.2541475775535105], [26.319493872472727,
       7.317529242917945], [29.025272664672727,
       12.005454001222029], [32.00921937712727,
       13.349956684511463], [35.29993109694545,
       5.352040089225467], [38.92894483829091,
       1.7491083700931236], [42.931039781927275,
       1.3841729422045932], [47.34457058660001,
       2.8991630437459444], [52.21183496636364,
       6.697350581717526], [57.57947905705454,
       8.868032437104791], [63.498944456109086,
       6.415711407919814], [70.02696122076362,
       5.917565966388773], [77.22609154872727,
       15.464441897660212], [85.16532935207273,
       46.097892319372605], [93.92076147050909,
       65.00210881747354], [103.5762968606,
       26.82979011574997], [114.22447074945455,
       12.196025528876234], [125.96733145956364,
       16.13244499672941], [138.91741840369093,
       44.14315341020613], [153.19884062274545,
       104.85862571326975], [168.94846620278182,
       189.8352169228327], [186.31723397021815,
       178.59561579235628], [205.47160003598182,
       190.31072745218344], [226.59513305192723,
       167.1645284032327], [249.89027346759997,
       86.39393455181767], [275.58027364783635,
       37.58798185821111], [303.9113374441091,
       29.789471117977268], [335.15497972505455,
       85.5436178148422], [369.6106284786545,
       367.4068125001968], [407.60849442385455,
       1275.225755645213], [449.5127356330182,
       3002.266636108816], [495.7249474937636,
       3451.533115637383], [546.6880114567272,
       1970.7562904186802], [602.8903394544,
       740.0963174535189], [664.8705546677091,
       613.4979772022668], [733.2226534998182,
       854.188246325088], [808.6016982268181,
       875.5314382539169], [891.7300948823455,
       860.6634704369059], [983.4045165408909,
       1196.0949821640443], [1084.5035383499455, 1369.158914959045]]

    gammac11_gammasp18_z99 = [[19.62371415969091, 0.11114613808565094], [21.641132498163635,
       0.22318920233125392], [23.865951776090906,
       0.2741149802484227], [26.319493872472727,
       0.3581740541338847], [29.025272664672727,
       0.5286267856638609], [32.00921937712727,
       0.5841114387297436], [35.29993109694545,
       0.30166041381773057], [38.92894483829091,
       0.1306017088894787], [42.931039781927275,
       0.11033811068671748], [47.34457058660001,
       0.2007232725583842], [52.21183496636364,
       0.3871850495730075], [57.57947905705454,
       0.4877993284934702], [63.498944456109086,
       0.3902192674043927], [70.02696122076362,
       0.37434279065094495], [77.22609154872727,
       0.7862623079814158], [85.16532935207273,
       1.8127451264332433], [93.92076147050909,
       2.396069696949687], [103.5762968606,
       1.2618611309918417], [114.22447074945455,
       0.7149343846841241], [125.96733145956364,
       0.9003288022048542], [138.91741840369093,
       1.9470199575042801], [153.19884062274545,
       3.8026269603733307], [168.94846620278182,
       6.101937066218874], [186.31723397021815,
       5.933871100944506], [205.47160003598182,
       6.351215776283767], [226.59513305192723,
       5.866064608583633], [249.89027346759997,
       3.6434193487102786], [275.58027364783635,
       2.001773781675013], [303.9113374441091,
       1.717006094110431], [335.15497972505455,
       3.851998787549225], [369.6106284786545,
       11.809009936793004], [407.60849442385455,
       32.24546861487769], [449.5127356330182,
       66.82017286161988], [495.7249474937636,
       76.16559060112496], [546.6880114567272,
       48.15342330104234], [602.8903394544,
       22.26450164596166], [664.8705546677091,
       19.583500548042185], [733.2226534998182,
       25.769393012599643], [808.6016982268181,
       26.715765363572505], [891.7300948823455,
       26.806990077169175], [983.4045165408909,
       35.230834073953694], [1084.5035383499455, 39.79512849932865]]

    ad_gammac1_z44 = [[12.367726508757421, 4.572243046321556*10**-8], [13.63919214777667,
       4.922447038634083*10**-8], [15.041370967591332,
       6.530216803758764*10**-8], [16.58770095277349,
       7.998510036928272*10**-8], [18.293001581533392,
       1.612463891418304*10**-7], [20.173615850372954,
       2.7209238641201575*10**-7], [22.247566899531275,
       3.412835448447508*10**-7], [24.53473074039463,
       2.3526545911253564*10**-7], [27.05702674013187,
       1.0694729125534346*10**-7], [29.838627689165882,
       9.883660884295661*10**-8], [32.906191464548634,
       1.0818976277088064*10**-7], [36.28911650971545,
       1.3766244259243824*10**-7], [40.01982357861175,
       1.4868018926291016*10**-7], [44.13406644481045,
       1.3355055189139007*10**-7], [48.67127455294964,
       1.625868696923608*10**-7], [53.674930896531,
       2.168504142122272*10**-7], [59.192988743563824,
       4.0548481445589616*10**-7], [65.27833120363529,
       5.276047716294315*10**-7], [71.98927804088372,
       4.1301740356775224*10**-7], [79.39014458995852,
       4.36404779047023*10**-7], [87.55185813135263,
       4.7094414955924204*10**-7], [96.5526376333376,
       7.012520071059242*10**-7], [106.47874337479462,
       1.292855794424257*10**-6], [117.42530363314877,
       1.3920915849512129*10**-6], [129.49722636001957,
       2.3151282401902424*10**-6], [142.810204581877,
       3.139877920032032*10**-6], [157.4918251609183,
       2.8651032453741363*10**-6], [173.68279154238627,
       2.759002069209858*10**-6], [191.5382722063983,
       2.3297616273146083*10**-6], [211.22938774771177,
       1.764619912963226*10**-6], [232.94485083479975,
       2.323422869605002*10**-6], [256.89277476512177,
       4.513660508416061*10**-6], [283.30266794918055,
       0.000010484445430619742], [312.42763343774516,
       0.000014779044408273449], [344.5467945717235,
       0.000015854549378042235], [379.9679700013215,
       0.000021852867566831194], [419.03062371071525,
       0.000021703609438901914], [462.10911832064687,
       0.00002328895255820392], [509.61630284690233,
       0.00002195829879121422], [562.0074692989141,
       0.00001935682243773189], [619.7847160369625,
       0.00002017157111621277], [683.5017597047544,
       0.000021689468044349526]]

    gammac1_gammasp18_z44 = [[12.367726508757421, 0.031117199823673756], [13.63919214777667,
       0.03340561082859777], [15.041370967591332,
       0.0487917958794043], [16.58770095277349,
       0.06273631010925741], [18.293001581533392,
       0.1652862624050908], [20.173615850372954,
       0.32949021272673196], [22.247566899531275,
       0.43296311125733034], [24.53473074039463,
       0.25277379135765693], [27.05702674013187,
       0.07763912382620994], [29.838627689165882,
       0.06654649485822436], [32.906191464548634,
       0.07291512022086541], [36.28911650971545,
       0.0994582971232782], [40.01982357861175,
       0.10683014709074272], [44.13406644481045,
       0.08774400631106534], [48.67127455294964,
       0.11221223366013969], [53.674930896531,
       0.16413304480224397], [59.192988743563824,
       0.3917303053091287], [65.27833120363529,
       0.5487136501671481], [71.98927804088372,
       0.3720052977619624], [79.39014458995852,
       0.3870809663371225], [87.55185813135263,
       0.41532931264255096], [96.5526376333376,
       0.7081188205649004], [106.47874337479462,
       1.5933642965154815], [117.42530363314877,
       1.7069864031329234], [129.49722636001957,
       3.2223867986604255], [142.810204581877,
       4.603900806026818], [157.4918251609183,
       4.024340119198074], [173.68279154238627,
       3.7460131279910738], [191.5382722063983,
       2.924270827941011], [211.22938774771177,
       1.9405438170872915], [232.94485083479975,
       2.7398365245675906], [256.89277476512177,
       6.353180796951084], [283.30266794918055,
       16.922370093084915], [312.42763343774516,
       24.4009601331479], [344.5467945717235,
       26.14067915835634], [379.9679700013215,
       36.46302796455312], [419.03062371071525,
       36.0703458066954], [462.10911832064687,
       38.66558029561192], [509.61630284690233,
       36.10538425385185], [562.0074692989141,
       31.169915257501582], [619.7847160369625,
       32.28793145369989], [683.5017597047544, 34.63585711080077]]

    gammac11_gammasp18_z44 = [[12.367726508757421, 0.0014041768628856913], [13.63919214777667,
       0.001514005245664142], [15.041370967591332,
       0.002053773609724553], [16.58770095277349,
       0.002533999497336025], [18.293001581533392,
       0.005394013111094231], [20.173615850372954,
       0.009419546500315294], [22.247566899531275,
       0.011916163837922832], [24.53473074039463,
       0.007953459501943547], [27.05702674013187,
       0.0033245438533888254], [29.838627689165882,
       0.0030330087548210457], [32.906191464548634,
       0.003318389798714779], [36.28911650971545,
       0.004275263620124581], [40.01982357861175,
       0.004612453666697448], [44.13406644481045,
       0.0040743201835337], [48.67127455294964,
       0.0050049563336473125], [53.674930896531,
       0.006797733932612226], [59.192988743563824,
       0.013391225448710525], [65.27833120363529,
       0.017704438719911647], [71.98927804088372,
       0.013434713612055026], [79.39014458995852,
       0.014147043597238293], [87.55185813135263,
       0.01524788510669293], [96.5526376333376,
       0.023382451358294715], [106.47874337479462,
       0.04490582920997797], [117.42530363314877,
       0.0483062643141049], [129.49722636001957,
       0.08195656589589397], [142.810204581877,
       0.11165363698207256], [157.4918251609183,
       0.10152903144263443], [173.68279154238627,
       0.09736302759020458], [191.5382722063983,
       0.08118926834561083], [211.22938774771177,
       0.05991852951760792], [232.94485083479975,
       0.08002184040190713], [256.89277476512177,
       0.15998343843954374], [283.30266794918055,
       0.3692164744031358], [312.42763343774516,
       0.5129074562099667], [344.5467945717235,
       0.5509890037771349], [379.9679700013215,
       0.7456693839385872], [419.03062371071525,
       0.7468004583125385], [462.10911832064687,
       0.8026546332873739], [509.61630284690233,
       0.7649846272235171], [562.0074692989141,
       0.6822813672176398], [619.7847160369625,
       0.7124435181799437], [683.5017597047544, 0.7665876298129901]]

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
    rcParams['legend.fontsize'] = 16

    plt.plot(z44_lims[:,0], z44_lims[:,1], linewidth=2.0, color='green', label='Adiabatic Spike, $\gamma_c = 1.0$')
    plt.plot(z44_lims[:,0], z44_lims[:,2], linewidth=2.0,label='$\gamma_{sp}=1.8$, $\gamma_c = 1.0$')
    plt.plot(z44_lims[:,0], z44_lims[:,3], linewidth=2.0, label='$\gamma_{sp}=1.8$, $\gamma_c = 1.1$')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('$\\frac{<\sigma v>}{<\sigma v>_{therm}}$')

    plt.axhline(1.0, linestyle='--', color='black', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('$\zeta = 0.44$')
    plt.ylim([10**-8, 10**5])
    plt.xlim([10**1, 1.2*10**3])

    plt.legend(loc=2)
    ax2 = fig.add_subplot(122)

    plt.plot(z99_lims[:,0], z99_lims[:,1], linewidth=2.0, color='green', label='Adiabatic Spike, $\gamma_c = 1.0$')
    plt.plot(z99_lims[:,0], z99_lims[:,2], linewidth=2.0,label='$\gamma_{sp}=1.8$, $\gamma_c = 1.0$')
    plt.plot(z99_lims[:,0], z99_lims[:,3], linewidth=2.0, label='$\gamma_{sp}=1.8$, $\gamma_c = 1.1$')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('$\\frac{<\sigma v>}{<\sigma v>_{therm}}$')

    plt.axhline(1.0, linestyle='--', color='black', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=2)
    plt.title('$\zeta = 0.9999$')
    plt.ylim([10**-8, 10**5])
    plt.xlim([10**1, 1.2*10**3])

    #plt.show()

def main():
    #theoryPlotter()
    #brazilPlot('plotsData/wideBoxResults.npy','plotsData/wideBoxBrazil.npy', 'brazil_wide_box.pdf')
    #brazilPlot('plotsData/narrowBoxResults.npy','plotsData/narrowBoxBrazil.npy', 'brazil_narrow_box.pdf')
    brazilPlot('plotsData/artificialBoxResults.npy','plotsData/artificialBoxBrazil.npy', 'brazil_artificial_box.pdf')
    #spectralPlot()
    #correlationPlot()
    #tsDistribution()
    #residmapComparison()
    dataModel()
    plt.show()

if __name__=='__main__':
    main()
