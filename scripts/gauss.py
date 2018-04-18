import numpy as np

sigma = 0.001
r = 10**np.linspace(-3.0, np.log10(180.0),200)
g = np.exp(-1.0*r**2/(2.0*sigma**2))/np.sqrt(2.0*np.pi*sigma**2)

file= open('gauss_01_degree.dat','wb')
for i in range(len(r)):
    file.write(str(r[i])+" " + str(g[i])+"\n")
file.close()
