'''
Ting BAO @ Tsinghua University
2022/07/26

A training task on Haldane model: 
1. Calculate the Berry curvature distribution in the BZ
2. Calculate the Berry phase
3. Calculate the sgn(M)-depedent Chern number
'''

from math import e, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_H(t1,t2,phi,k,M):
    '''t1 for NN hopping, 
    t2 for NNN hopping,
    phi for phase in NNN hopping,
    return Hamiltonian matrix as k=[kx,ky] under given parameter
    '''

    # NN: neareat neighbour, mark each element a_i 
    # NNN: next nearest neighbour, mark each element b_i
    NN = np.array([[0.5,sqrt(3)/6],[-0.5,sqrt(3)/6],[0,-sqrt(3)/3]])
    NNN = np.array([[1.,0.],[0.5,sqrt(3)/2],[0.5,-sqrt(3)/2]])

    H12=t1*(sum([e**(-1j*np.dot(k,delta)) for delta in NN]))
    H21=t1*(sum([e**(1j*np.dot(k,delta)) for delta in NN]))
    

    H=np.array([[H11,H12],[H21,H22]])

    return H



def main(t1=1,t2=0.1,M=0.1,phi=pi/8):
    '''parameters, change as required
    '''

    # do k-mesh
    sample_kx=np.arange(-1,1.05,0.05)
    sample_ky=np.arange(-1,1.05,0.05)
    xx,yy=np.meshgrid(sample_kx,sample_ky)
    kx=xx.flatten()
    ky=yy.flatten()
    # each (kx[i],ky[i]) mark a k sample point in the BZ

    get_H(t1=1,t2=0.1,phi=pi/8,k=[1.,1.])
    print(get_H(t1=1,t2=0.1,phi=pi/8,k=[1.,1.]))
    print("OK")

if __name__=='__main__':
    main()

