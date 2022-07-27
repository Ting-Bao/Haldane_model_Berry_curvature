import numpy as np

a=np.array([[1,2],[3,4]])
b=np.array([[-1,0]])
b=np.array([[-1,0]])
print(b)
print(np.dot(np.dot(b,a),b.T))
print(b)

'''
sample_kx=np.arange(-1,1.05,0.05)
sample_ky=np.arange(-1,1.05,0.05)
xx,yy=np.meshgrid(sample_kx,sample_ky)
kx=xx.flatten()
ky=yy.flatten()
'''