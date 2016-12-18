import numpy as np
from scipy.linalg import orth

N = 160 #Size of the image
M = int(0.4*N**2) #Maximum number of measurements

Phi = np.random.randn(M,N**2)
Phi_or = orth(Phi.transpose()).transpose()

np.save('map_single_pixel/Phi_g_'+str(N),Phi_or)