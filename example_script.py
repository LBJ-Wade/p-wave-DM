print "initializing..."
import math
import numpy as np
import pyfits
#from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
#from astropy import wcs
from BinnedAnalysis import *
from SummedLikelihood import *
import pyLikelihood as pyLike
from gt_apps import evtbin, srcMaps, gtexpcube2

print "Done!"

        
def correlation_calculation():     
    sourcemap = 'srcmap.fits'
            
    obs = BinnedObs(srcMaps=sourcemap, expCube='ltcube.fits', binnedExpMap='exposure.fits', irfs='CALDB')
    like = BinnedAnalysis(obs, 'xmlmodel.xml', optimizer='MINUIT')
    like.tol=1e-8
    like_obj = pyLike.Minuit(like.logLike)
    like.fit(verbosity=3,optObject=like_obj)
    like.writeXml('xmlmodel.xml')
    
    like1 = BinnedAnalysis(obs, 'xmlmodel.xml', optimizer='DRMNFB')
    like1.thaw(like1.par_index('Disk Component','Index'))
    like1.thaw(like1.par_index('Disk Component','Prefactor'))
    like1.thaw(like1.par_index('Box Component','Normalization'))
    like1.tol=1e-5
    like1obj = pyLike.Minuit(like1.logLike)
    like1.fit(verbosity=0,optObject=like1obj, covar=False)

    like1.writeXml('xmlmodel.xml')
        
    like2 = BinnedAnalysis(obs, 'xmlmodel.xml', optimizer='NewMinuit')
    like2.tol=1e-8
    like2obj = pyLike.Minuit(like1.logLike)
    like2.fit(verbosity=3,optObject=like1obj, covar=True)
        
    cov = like2.covariance
    corr = cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])
    corr2 = cov[0][2]/np.sqrt(cov[0][0]*cov[2][2])
    print "Correlations:"
    print corr
    print corr2
    

def main():
    correlation_calculation()
    
if __name__=='__main__':
    main()