import math
import numpy as np
from scipy.special import gammainc
import pickle
import pylab as plt
import pyfits
from math import sin, cos, asin, acos, radians
from astropy import wcs


#Strictly finds the Poisson upper limit (problematic for large counts)
def upper_limit(N,conf,b):
    #Calculates upper limit on signal s, given background b, number of events N, and confidence interval conf (95% = 0.05)
    #Decides whether the value is really big, in which case do everything with ints. Otherwise, use floats
    #First, calculate denominator
    denom=0.
    for m in range(0,N+1):
            if m>10:
                denom+=(b*math.e/m)**m/math.sqrt(2*math.pi*m)
            else:
                denom+=b**m/math.factorial(m)
    s = 0.
    numer=denom
    while math.exp(-1.0*s)*numer/denom>conf:
        #Calculate numerator
        numer=0.0
        for m in range(0,N+1):
            if m>10:
                numer+=((s+b)*math.e/m)**m/math.sqrt(2*math.pi*m)
            else:
                numer+=(s+b)**m/math.factorial(m)
    
        s+= d_box_flux
    print "Upper limit is " + str(s)
    return s
    
def factorial(x):
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
    thesum = 0.0
    for i in range(len(g)-1):
        thesum += 0.5*(g[i]+g[i+1])*(x[i+1]-x[i])
    return thesum

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
    
num_ebins = 51
energies = 10**np.linspace(np.log10(300),np.log10(300000),num_ebins)
ebin_widths = np.diff(energies)
    
#We need to account for exposure. How to do this?
#We have counts in each bin, model for counts in each bin
#Also have model of box- in flux
#Need to multiply box by value of exposure in each bin, to get expected number of counts

#What's the exposure as a function of energy?
exposure_file = pyfits.open('very_center_exposure.fits')
exp_energies = np.zeros((len(exposure_file[1].data)))
for i in range(len(exp_energies)):
    exp_energies[i] = float(exposure_file[1].data[i][0])
def get_exposure(E):
    index = np.argmin(np.abs(exp_energies-E))
    return exposure_file[0].data[index][52][52]

exposures = np.zeros(len(energies))
for i in range(len(energies)):
    exposures[i] = get_exposure(energies[i])

#Load the results of the Science Tools fit, and set up arrays for later
#counts_file = open('very_center_counts.pk1','rb')
#counts_file = open('very_center_ccube.pk1','rb')
#counts = pickle.load(counts_file)
#counts_file.close()

counts_cube_file = pyfits.open('very_center_ccube.fits')
counts = np.zeros((50,10,10))
for i in range(50):
    for j in range(10):
        for k in range(10):
            counts[i,j,k] = float(counts_cube_file[0].data[i][j][k])

#Calculating an upper limit, using a likelihood ratio test
def likelihood_upper_limit(counts,model_counts):
    box_flux = np.zeros((num_ebins))
    for i in range(num_ebins):        
        box_counts = box_counts_matrix[i,:]*exposures/10.**11.
        if counts[i] != 0:
            d_box_flux = 0.01*counts[i]/(ebin_widths[i])
        else:
            d_box_flux = 0.1/ebin_widths[i]
        #Find minimum i.e. best fit value
        while loglikelihood(counts,model_counts,(box_flux[i]+d_box_flux)*box_counts)<loglikelihood(counts,model_counts,box_flux[i]*box_counts):
            box_flux[i] += d_box_flux
        box_significance = sigma_given_p(pvalue_given_chi2(loglikelihood(counts,model_counts,box_flux[i]*box_counts),num_ebins-1))
        while box_significance<2.0:
            box_significance = sigma_given_p(pvalue_given_chi2(loglikelihood(counts,model_counts,box_flux[i]*box_counts),num_ebins-1))
            box_flux[i] += d_box_flux
        box_flux[i] *= energies[i]/10.**11.
        print "upper limit in bin " + str(i) + " is " + str(box_flux[i])
    return box_flux


#input: cubes with dimensions (50,10,10)
def loglikelihood2(counts,model_counts,box):
    f0 = 0.0
    f1 = 0.0
    #i loops over energy slices
    for i in range(len(counts)):
        #j loops over ra
        for j in range(10):
            #k loops over dec
            for k in range(10):
                #m = number of counts
                m = counts[i,j,k]
                #null hypothesis counts:
                mu0 = model_counts[i,j,k]
                #background+signal counts
                mu1 = model_counts[i,j,k]+box[i,j,k]
                #Do we need stirling's approximation?   
                if m>20:	
                    f0 += m-mu0+m*np.log(mu0/m)-0.5*np.log(m)-np.log(np.sqrt(2*np.pi))
                    f1 += m-mu1+m*np.log(mu1/m)-0.5*np.log(m)-np.log(np.sqrt(2*np.pi))
                else:
                    f0 += np.log((mu0**m)*np.exp(-1.0*mu0)/factorial(m))
                    f1 += np.log((mu1**m)*np.exp(-1.0*mu1)/factorial(m))
    return 2*(f0-f1)	

#Returns dim (50,10,10) array with unity counts smeared out by PSF
#To get counts as a function of energy given a flux, just multiply this array by the box counts
def make_box_cube():
    box_cube = np.zeros((50,10,10))
    for i in range(50):
        ang_res = psf(energies[i])
        #j loops over ra
        for j in range(10):
            #k loops over dec
            for k in range(10):
                #Here we calculate the integrated value of a 2d gaussian function
                sigma = ang_res/np.sqrt(2)
                xvalue = 1.0/np.sqrt(2.0*np.pi*sigma**2)*np.exp((-1.0*(0.1*(j-5)+0.05)**2)/(2*sigma**2))
                yvalue = 1.0/np.sqrt(2.0*np.pi*sigma**2)*np.exp((-1.0*(0.1*(k-5)+0.05)**2)/(2*sigma**2))
                #integral here is just area (0.1 deg * 0.1 deg)* value of function at the center of each pixel
                box_cube[i,j,k] = xvalue*yvalue*0.01
    #Visualization of the cube, if desired
    #plt.imshow(box_cube[25,:,:])
    #plt.show()
    return box_cube

    
#A function to get the upper limit, given a counts cube and source map cube
#Dimensions of the cube are ra/dec, and energy
def likelihood_upper_limit2(c_cube,srcmap,confidence):
    #Array to hold results of flux upper limit calculation
    box_flux = np.zeros((num_ebins))
    #Loop through upper edge of box
    box_cube = make_box_cube()
    for i in range(num_ebins-1): 
        print "Calculating upper limit in bin " + str(i) + " which is at energy " + str(energies[i]) + " MeV"     
        #Calculate number of expected counts from the box, in each energy bin
        #this is found via convolving the pure box with the energy dispersion
        #then integrating in each energy bin
        #god help you if you are trying to figure out what's happening here. hopefully it works
        #needs more comments in the future
        x_fine_grid = np.linspace(0.0, 300000, 300000)
        pure_box = np.concatenate([1.0+np.zeros((int(np.argmin(np.abs(x_fine_grid-energies[i]))))),np.zeros((300000-int(np.argmin(np.abs(x_fine_grid-energies[i])))))])
        sigma = e_res(energies[i])*energies[i]
        dispersion = blur(np.linspace(0,6*sigma,6*sigma),3*sigma,sigma)
        convolved_pure_box = np.convolve(pure_box, dispersion,'same')
        #now that we have a convolved box, figure out what the number of counts in each energy bin equates to
        #two pieces- the integrated number of counts per bin, and the relative exposure per bin
        box_counts = np.zeros((num_ebins-1))
        for j in range(num_ebins-1):
            xlow = np.argmin(np.abs(x_fine_grid-energies[j]))
            xhigh = xlow+ebin_widths[j]
            #Exposure correction. Factor of 10^11 is just to make things more convenient
            box_counts[j] = get_integral(x_fine_grid[xlow:xhigh],convolved_pure_box[xlow:xhigh])*exposures[j]/10.**11./10**3
        #Need to turn the total counts into a counts cube
        box_counts_cube = multiply_multidimensional_array(box_counts,box_cube)
        box_counts_cube *= 1.0/sum(sum(sum(box_counts_cube[:,:,:])))
        print "total number of counts in slice " + str(i) + " is " + str(sum(sum(c_cube[i,:,:])))
        print "total number of model counts in slice " + str(i) + " is " + str(sum(sum(srcmap[i,:,:])))
        #The increment should be equivalent to a few counts in the energy slice we care about
        #say, 5 counts
        #Find minimum i.e. best fit value
        d_box_flux = (sum(sum(c_cube[i,:,:]))+sum(sum(srcmap[i,:,:])))*0.5/100.
        print "dbox flux = " + str(d_box_flux)
        print "sum = " + str(1.0*sum(sum(box_counts_cube[i,:,:])))
        print "finding minimum..."
        while loglikelihood2(c_cube,srcmap,(box_flux[i]+d_box_flux)*box_counts_cube)<loglikelihood2(c_cube,srcmap,box_flux[i]*box_counts_cube):
            box_flux[i] += d_box_flux
                
        print "finding upper limit..."
        box_significance = sigma_given_p(pvalue_given_chi2(loglikelihood2(c_cube,srcmap,box_flux[i]*box_counts_cube),(num_ebins-1)))
        
        while box_significance<2.0:
            box_flux[i] += d_box_flux
            box_significance = sigma_given_p(pvalue_given_chi2(loglikelihood2(c_cube,srcmap,box_flux[i]*box_counts_cube),(num_ebins-1)))
            print "integrated counts = " + str(box_flux[i]*sum(sum(sum(box_counts_cube[:,:,:]))))
            print box_significance
            
        #Return the integrated flux
        box_flux[i] *= 1.0/exposures[i]
        print "upper limit in bin " + str(i) + " is " + str(box_flux[i])
    return box_flux

  
#Commented out code generates the box matrix, which gives a blurred box spectrum  
"""
print "Generating matrix..."
#Save time and just calculate box edges once
box_counts_matrix = np.zeros((len(energies),len(energies)))
print energies
print len(energies)
print ebin_widths
print len(ebin_widths)
i = 0
for energy in energies:
    box_counts_matrix[i,:] = generate_counts(energy,energies)
    i += 1
file = open('box_matrix.pk1','wb')
pickle.dump(box_counts_matrix,file)
file.close()
print "Done!"

fig = plt.figure()
ax = fig.add_subplot(111)
the_box = np.concatenate([1.0+np.zeros((500)),np.zeros((500))])
ax.plot(np.linspace(0,100000,1000),the_box,color='black',linewidth=3,label='Simple Box')
ax.plot(np.linspace(0,100000,1000)[5:],convolve(np.linspace(0,100000,1000),the_box)[5:],color='blue',linewidth=3,label='Convolved Box')
plt.xlabel('Energy [eV]')
plt.legend()
plt.show()
raw_input('wait for key')
"""

model_counts = np.zeros((num_ebins-1))
model_counts2 = np.zeros((num_ebins))
model_counts3 = np.zeros((num_ebins))

#Moderate Model
model_file = open('very_center_model_moderate.pk1','rb')
model = pickle.load(model_file)
model_file.close()

#Using the source maps to make model cubes
srcmodel_file = pyfits.open('very_center_srcmap.fits')
#The 4 different components of the model
source_map = np.zeros((4,50,10,10))
for h in range(len(model)):
    for i in range(50):
        for j in range(10):
            for k in range(10):
                source_map[h,i,j,k] = srcmodel_file[h+3].data[i][j][k]

srcmodel_file.close()

model_counts = np.zeros((num_ebins-1))
moderate_model_cube = np.zeros((num_ebins-1,10,10))
#Combine the various fits from gt_apps into a single fit model
for h in range(len(model)):
    for i in range(num_ebins-1):
        for j in range(10):
            for k in range(10):
                norm_factor = model[h][i]/sum(sum(source_map[h,i,:,:]))
                #model_counts[i] += model[i][h]
                moderate_model_cube[i,j,k] += norm_factor*float(source_map[h,i,j,k])
                
        
moderate_upper_limits = likelihood_upper_limit2(counts,moderate_model_cube,2.0)

"""
#Conservative model
model_file = open('very_center_model_conservative.pk1','rb')
model = pickle.load(model_file)
model_file.close()

#Combine the various fits from gt_apps into a single fit model
for i in range(num_ebins):
    for j in range(len(model)):
        model_counts[i] += model[j][i]
        model_counts3[i] += model[j][i]
        
print "conservative model counts: " + str(model_counts)
conservative_upper_limits = likelihood_upper_limit(counts,model_counts)
print "actual counts: " + str(counts)

#Aggressive model
model_file = open('very_center_model_aggressive.pk1','rb')
model = pickle.load(model_file)
model_file.close()
model_counts = np.zeros((num_ebins))
#Combine the various fits from gt_apps into a single fit model
for i in range(num_ebins):
    for j in range(len(model)):
        model_counts[i] += model[j][i]
print "aggressive model counts: " + str(model_counts)
aggressive_upper_limits = likelihood_upper_limit(counts,model_counts)
"""
make_brazil_bands = False
if make_brazil_bands:
    #Poisson randomization of the counts in each bin, so that we get 95% and 68% containment for the upper limit
    trials = 50
    mc_limits = np.zeros((trials,num_ebins))
    for i in range(trials):
        print "Trial " + str(i)
        the_counts = np.random.poisson(np.random.poisson(model_counts))
        mc_limits[i,:] = likelihood_upper_limit(the_counts,model_counts)
    
    lower_95 = np.zeros((num_ebins))        
    lower_68 = np.zeros((num_ebins))        
    upper_95 = np.zeros((num_ebins))        
    upper_68 = np.zeros((num_ebins))   
    median = np.zeros((num_ebins))     
    for i in range(num_ebins):
        lims = mc_limits[:,i]
        lims.sort()
        lower_95[i] = lims[int(0.025*trials)]
        upper_95[i] = lims[int(0.975*trials)]
        lower_68[i] = lims[int(0.15865*trials)]
        upper_68[i] = lims[int(0.84135*trials)]
        median[i] = lims[int(0.5*trials)]

"""
#Plotting the results
fig=plt.figure()
ax = fig.add_subplot(111)
#ax.plot(energies, counts, marker='s', color='black', linewidth=0)
#ax.plot(energies, 10**linear_fit(np.log10(energies)),color='blue')
ax.plot(energies[2:45], conservative_upper_limits[2:45], marker='s',markersize=5, color='black', linewidth=1.0,label='Conservative Upper Limit')
ax.plot(energies[2:45], moderate_upper_limits[2:45], marker='s',markersize=5, color='red', linewidth=1.0,label='Moderate Upper Limit')
ax.plot(energies[2:45], aggressive_upper_limits[2:45], marker='s',markersize=5, color='blue', linewidth=1.0,label='Aggressive Upper Limit')

plt.ylabel('Flux Upper Limit')
plt.xlabel('Energy [MeV]')
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(loc=1)
plt.show()"""
        
fig = plt.figure()
ax2 = fig.add_subplot(111)
#ax2.plot(energies, counts, marker='s', color='black', linewidth=0,label='Data')
#ax2.plot(energies,model_counts,color='black',label='Conservative Model')
#ax2.plot(energies,model_counts2,color='blue',label='Aggressive Model')
ax2.plot(energies,moderate_upper_limits,color='red',label='Moderate Model')

ax2.legend(loc=1)
ax2.set_yscale('log')
ax2.set_xscale('log')
plt.show()

#ax2.fill_between(energies[2:45],lower_95[2:45],upper_95[2:45], where=upper_95[2:45]> lower_95[2:45], facecolor='yellow',label='95 Percent Containment')
#ax2.fill_between(energies[2:45],lower_68[2:45],upper_68[2:45], where=upper_68[2:45]> lower_68[2:45], facecolor='green',label='68 Percent Containment')
#ax2.plot(energies[2:45], conservative_upper_limits[2:45], marker='s',markersize=5, color='black', linewidth=1.0,label='Conservative Upper Limit')
plt.show()

if make_brazil_bands:
    #Residuals plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(energies[2:45], (conservative_upper_limits[2:45]-median[2:45])/median[2:45], marker='s',markersize=5, color='black', linewidth=1.0,label='Upper Limit Residuals [(Data-Model)/Model]')
    ax.fill_between(energies[2:45],(lower_95[2:45]-median[2:45])/median[2:45],(upper_95[2:45]-median[2:45])/median[2:45], where=(upper_95[2:45]-median[2:45])/median[2:45]> (lower_95[2:45]-median[2:45])/median[2:45], facecolor='yellow',label='95 Percent Containment')
    ax.fill_between(energies[2:45],(lower_68[2:45]-median[2:45])/median[2:45],(upper_68[2:45]-median[2:45])/median[2:45], where=(upper_68[2:45]-median[2:45])/median[2:45]> (lower_68[2:45]-median[2:45])/median[2:45], facecolor='green',label='68 Percent Containment')
    plt.ylabel('Relative Upper Limit [Counts/MeV]')
    plt.xlabel('Energy [MeV]')
    ax.set_xlim(400, 250000)
    ax.axhline(0.0,linewidth=1,color='black',linestyle='--')
    ax.set_xscale('log')
    ax.legend(loc=1)
    plt.show()
    ax2 = fig.add_subplot(122)
    ax2.errorbar(energies, (counts-model_counts)/model_counts,xerr = 0, yerr=np.sqrt(counts-model_counts)/model_counts, marker='s', color='black', linewidth=0,label='Residuals [(Data-Model)/Model]')
    ax.axhline(0.0,linewidth=1,color='black',linestyle='--')
    ax2.legend(loc=1)
    ax2.set_xscale('log')
    plt.show()
