import numpy as np
sample_kx=np.arange(-1,1.05,0.05)
sample_ky=np.arange(-1,1.05,0.05)
xx,yy=np.meshgrid(sample_kx,sample_ky)
kx=xx.flatten()
ky=yy.flatten()