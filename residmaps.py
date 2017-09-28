import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from scipy.special import gammainc
from scipy.misc import factorial
import math
import pyfits
import pickle
from BinnedAnalysis import *
import matplotlib.colors as colors
from matplotlib.pyplot import rc
from matplotlib import rcParams

from astropy.visualization.wcsaxes.frame import EllipticalFrame

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.signal import convolve2d

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
    

def make_model_cubes():
    models = ['few_src']#, '3fgl_disk', '1fig', '3fgl']
    fs_srcmap = pyfits.open('few_sources_srcmap.fits')
    file = open('models/high_tol_results.pk1','rb')
    g = pickle.load(file)
    file.close()

    results = {}
    my_arr = {}
    for model in models:
        my_arr[model] = np.zeros((28, 50, 50))
        results[model]={}
        sources = g[model]
        for source in sources:
            results[model][source] = np.zeros((28, 50, 50))
            i = 3
            while fs_srcmap[i].header['EXTNAME'] != source[:-1] and fs_srcmap[i].header['EXTNAME'] != source:
                i += 1
            for e_bin in np.arange(0, 25):
                if g[model][source][e_bin]>0.0:
                    results[model][source][e_bin,:,:] = fs_srcmap[i].data[e_bin]*g[model][source][e_bin]/sum(sum(fs_srcmap[i].data[e_bin]))

        for source in results[model].keys():
            my_arr[model] += results[model][source]
    file = open('result_cube.pk1','wb')
    pickle.dump(my_arr,file)
    file.close()
    return results

#Make a residual map given a particular model (from the function make_model_cubes)
def make_ROI_map(type):
    
    obs_complete = BinnedObs(srcMaps='/Users/christian/physics/p-wave/6gev/6gev_srcmap_03.fits', expCube='/Users/christian/physics/p-wave/6gev/6gev_ltcube.fits', binnedExpMap='/Users/christian/physics/p-wave/6gev/6gev_exposure.fits', irfs='CALDB')
    
    #Flucuate the window data
    like = BinnedAnalysis(obs_complete, 'xmlmodel_free.xml', optimizer='NEWMINUIT')
    like.tol=1e-10
    like_obj = pyLike.Minuit(like.logLike)
    likelihood = like.fit(verbosity=3,optObject=like_obj)
    like.writeXml('xmlmodel_free.xml')

    f = pyfits.open('6gev_srcmap_03.fits')
    my_arr = np.zeros((50,50))
    for source in like.sourceNames():
        for j in range(3,9):
            if source == f[j].header['EXTNAME']:
                the_index = j
        for bin in range(50):
            num_photons = like._srcCnts(source)[bin]
            model_counts = num_photons*f[the_index].data[bin]/np.sum(np.sum(f[the_index].data[bin]))
            my_arr += model_counts
    f.close()
    print "likelihood = " + str(likelihood)

    image_data = fits.getdata('6gev_image.fits')
    filename = get_pkg_data_filename('6gev_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=[15,10])
    """
    ax=fig.add_subplot(131,projection=wcs)
    plt.scatter([359.9442], [-00.0462], color='black',marker='x',s=45.0,transform=ax.get_transform('world'))
    l, b = ra_dec_to_l_b(266.3434922, -29.06274323)
    plt.scatter([l], [b], color='black',marker='x',s=45.0,transform=ax.get_transform('galactic'))
    l, b = ra_dec_to_l_b(267.1000722, -28.27707114)
    plt.scatter([l], [b], color='black',marker='x',s=45.0,transform=ax.get_transform('galactic'))
    l, b = ra_dec_to_l_b(266.5942898, -28.86244442)
    plt.scatter([l], [b], color='black',marker='x',s=45.0,transform=ax.get_transform('galactic'))
    c = Wedge((gc_l, gc_b), 15.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)

    mappable = plt.imshow(my_arr,cmap='inferno',interpolation='bicubic',origin='lower',norm=colors.PowerNorm(gamma=0.6))

    cb = plt.colorbar(mappable,label='Counts per pixel')
    mappable.set_clip_path(ax.coords.frame.patch)
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    ax.grid(color='white',ls='dotted')

    ax=fig.add_subplot(132,projection=wcs)
    c = Wedge((gc_l, gc_b), 15.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)
    mappable = plt.imshow(image_data,cmap='inferno',interpolation='bicubic',origin='lower',norm=colors.PowerNorm(gamma=0.6))
    cb = plt.colorbar(mappable,label='Counts per pixel')
    mappable.set_clip_path(ax.coords.frame.patch)
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    ax.grid(color='white',ls='dotted')
    """
    resid = image_data-my_arr
    resid_sigma = np.zeros((len(resid.ravel()), 1))
    model_array = my_arr.ravel()
    for q in range(len(resid_sigma)):
        resid_sigma[q] = frequentist_counts_significance(float(resid.ravel()[q]), float(model_array[q]))
    resid_sigma = np.reshape(resid_sigma,[50,50])

    """
    plt.scatter([359.9442], [-00.0462], color='black',marker='x',s=45.0,transform=ax.get_transform('world'))
    l, b = ra_dec_to_l_b(266.3434922, -29.06274323)
    plt.scatter([l], [b], color='black',marker='x',s=45.0,transform=ax.get_transform('galactic'))
    l, b = ra_dec_to_l_b(267.1000722, -28.27707114)
    plt.scatter([l], [b], color='black',marker='x',s=45.0,transform=ax.get_transform('galactic'))
    l, b = ra_dec_to_l_b(266.5942898, -28.86244442)
    plt.scatter([l], [b], color='black',marker='x',s=45.0,transform=ax.get_transform('galactic'))
    """
    kernel = np.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0], [1.0, 1.0,1.0]])/9.0
    
    ax=fig.add_subplot(111,projection=wcs)
    ax=plt.gca()
    
    c = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)
    if type == 'Data':
        mappable=plt.imshow(image_data,cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='bicubic')#
        cb = plt.colorbar(mappable,label='Counts per pixel')
        mappable.set_clip_path(ax.coords.frame.patch)
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        ax.grid(color='white',ls='dotted')
        plt.savefig('plots/6gev_ROI_03.pdf',bbox_inches='tight')
        plt.show()
        
    if type == 'Resid':
        resid = image_data-my_arr
        mappable=plt.imshow(resid, cmap='seismic',origin='lower', vmin=-20, vmax=20, interpolation='bicubic')#norm=colors.SymLogNorm(linthresh=5,linscale=1.0),
        cb = plt.colorbar(mappable,label='Counts per pixel')
        mappable.set_clip_path(ax.coords.frame.patch)
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        ax.grid(color='black',ls='dotted')
        plt.savefig('plots/6gev_resid_03.pdf',bbox_inches='tight')
        
    if type == 'Sigma':
        mappable=plt.imshow(resid_sigma,cmap='seismic',origin='lower',vmin=-5.0,vmax=5.0, interpolation='bicubic')#norm=colors.SymLogNorm(linthresh=5,linscale=1.0),
        cb = plt.colorbar(mappable,label='Counts per pixel')
        mappable.set_clip_path(ax.coords.frame.patch)
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        ax.grid(color='white',ls='dotted')
        plt.savefig('plots/6gev_sigma_03.pdf',bbox_inches='tight')
        
    if type == 'Model':
        mappable=plt.imshow(my_arr,cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6), vmin=0, vmax=65, interpolation='bicubic')#
        cb = plt.colorbar(mappable,label='Counts per pixel')
        mappable.set_clip_path(ax.coords.frame.patch)
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        ax.grid(color='white',ls='dotted')
        plt.savefig('plots/6gev_model_03.pdf',bbox_inches='tight')
    
        

#0.04 degrees: 15798.5084759
#0.03 degrees: 15797.2262217
#0.02 degrees: 15803.1695199
#0.01 degrees: 15808.9181639
#pt source: 15808.1352914
#code to make a python list of dictionaries of 3fgl sources

def make_fgl_pk1():
    g = pyfits.open('/Users/christian/physics/PBHs/3FGL.fits')
    j = []
    for entry in g[1].data:
        if entry['SpectrumType'].split(' ')[0]=='LogParabola':
            st = 'LogParabola'
        if entry['SpectrumType'].split(' ')[0] == 'PowerLaw':
            st= 'PL'
        if entry['SpectrumType'].split(' ')[0] =='PLExpCutoff':
            st = 'PE'
        if entry['SpectrumType'].split(' ')[0] == 'PLSuperExpCutoff':
            st = 'SE'
        j.append({'src_name':entry['Source_Name'], 'L':entry['GLON'], 'B':entry['GLAT'], 'SpectrumType':st})
    file = open('fgl.pk1','wb')
    pickle.dump(j,file)
    file.close()
    print "Done!"

#Code to make a cool plot that overlays the catalog source locations on the data
def make_overlay_plot(catalog):
    #catalog = '3fgl' or '1fig'

    if catalog=='1fig':
        file = open('fig.pk1','rb')
        g=  pickle.load(file)
        file.close()
    if catalog == '3fgl':
        file = open('fgl.pk1','rb')
        g=  pickle.load(file)
        file.close()
    if catalog == 'few_src':
        file = open('few_src.pk1', 'rb')
        g = pickle.load(file)
        file.close()

    image_data = fits.getdata('10gev_image.fits')
    filename = get_pkg_data_filename('10gev_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[15,10])
    plt.subplot(projection=wcs)
    ax=plt.gca()
    ax.grid(color='white',ls='dotted')

    n_lp = 0
    n_pl = 0
    n_pe = 0
    n_se = 0

    for entry in g:
        if entry['L']>180.0:
            dist = np.sqrt((entry['L']-gc_l)**2+(entry['B']-gc_b)**2)
        elif entry['L']<180:
            dist = np.sqrt((entry['L']-(360.0-gc_l))**2+(entry['B']-gc_b)**2)

        if dist<1.0:
            if entry['SpectrumType']=='LogParabola' and n_lp ==0:
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='#49ff00',marker='x',s=45.0,transform=ax.get_transform('world'),label='LogParaboloa Sources')
                n_lp+=1
            elif entry['SpectrumType']=='LogParabola':
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='#49ff00',marker='x',s=45.0,transform=ax.get_transform('world'))
                n_lp+=1

            if entry['SpectrumType'] =='PowerLaw' and n_pl ==0:
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='c',marker='x',s=45.0,transform=ax.get_transform('world'),label='PowerLaw Sources')
                n_pl +=1
            elif entry['SpectrumType'] =='PowerLaw':
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='c',marker='x',s=45.0,transform=ax.get_transform('world'))
                n_pl +=1

            if entry['SpectrumType']=='PLSuperExpCutoff' and n_se ==0:
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='#b8b8b8',marker='x',s=45.0,transform=ax.get_transform('world'),label='PLSuperExpCutoff Sources')
                n_se+=1
            elif entry['SpectrumType']=='PLSuperExpCutoff':
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='#b8b8b8',marker='x',s=45.0,transform=ax.get_transform('world'))
                n_se+=1

            if entry['SpectrumType'] =='PLExpCutoff' and n_pe ==0:
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='#e3ff00',marker='x',s=45.0,transform=ax.get_transform('world'),label='PLExpCutoff Sources')
                n_pe +=1
            elif entry['SpectrumType'] =='PLExpCutoff':
                ax.scatter([float(entry['L'])], [float(entry['B'])], color='#e3ff00',marker='x',s=45.0,transform=ax.get_transform('world'))
                n_pe +=1

    ax.scatter([359.9442], [-00.0462], color='black',marker='x',s=45.0,transform=ax.get_transform('world'))
    c = Wedge((gc_l, gc_b), 15.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    ax.add_patch(c)
    mappable = plt.imshow(image_data,cmap='inferno',interpolation='bicubic',origin='lower',norm=colors.PowerNorm(gamma=0.6))
    cb = plt.colorbar(mappable,label='Counts per pixel')
    """
    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    if catalog=='1fig':
        text1 = leg.get_texts()
        print text1[0]
        dir(text1[0])
        text1[0].set_color('white')
    if catalog=='3fgl':
        text1, text2 = leg.get_texts()
        text1.set_color('white')
        text2.set_color('white')
    """

    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    #plt.savefig('plots/'+catalog+'_overlay.png',bbox_inches='tight')
    plt.show()

def main():
    make_ROI_map('Sigma')

if __name__=='__main__':
    main()