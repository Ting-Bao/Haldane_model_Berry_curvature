'''
Ting BAO @ Tsinghua University
2022/07/26

A training task on Haldane model: 
1. Calculate the Berry curvature distribution in the BZ
2. Calculate the Berry phase
3. Calculate the sgn(M)-depedent Chern number
'''

from math import cos, e, pi, sqrt
from os import pathsep
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
    NN = np.array([[0.5,sqrt(3)/6],[-0.5,sqrt(3)/6],[0.,-sqrt(3)/3]])
    NNN = np.array([[-0.5,sqrt(3)/2],[-0.5,-sqrt(3)/2],[1.,0.]])

    H11=2*t2*(sum([cos(phi+np.dot(k,bi)) for bi in NNN]))+M
    H12=t1*(sum([e**(-1j*np.dot(k,ai)) for ai in NN]))
    H21=t1*(sum([e**(1j*np.dot(k,ai)) for ai in NN]))
    H22=2*t2*sum([cos(phi-np.dot(k,bi)) for bi in NNN])-M

    H=np.array([[H11,H12],[H21,H22]])
    return H

def get_berryphase(kxlist,kylist,berry_curv,dsize=0.05*0.05):
    """ intergrate in a rectangle area to get the Berry phase 
    dsize means the step using in meshing the BZ
    the length of recipical lattice vector is 4*pi/sqrt(3)
    """
    klength=4*pi/sqrt(3)
    phase=0.
    for i in range(len(kxlist)):
        kx=kxlist[i]
        ky=kylist[i]
        if kx>=0 and kx<klength*sqrt(3)/2 and ky>=-(1/sqrt(3))*kx and ky<-(1/sqrt(3))*kx+klength:
            phase+=dsize*berry_curv[i]
    print(phase)
    return phase


def main(t1,t2,M,phi):
    '''parameters, change as required
    '''

    # do k-mesh
    sample_kx=np.arange(-8.,8.1,0.05)
    sample_ky=np.arange(-8.,8.1,0.05)
    xx,yy=np.meshgrid(sample_kx,sample_ky)
    kxlist=xx.flatten()
    kylist=yy.flatten()
    # each (kx[i],ky[i]) mark a k sample point in the BZ

    berry_curv_1=[]
    berry_curv_2=[]
    berry_curv_total=[]

    for i in range(len(kxlist)):
        kx=kxlist[i]
        ky=kylist[i]
        H=get_H(t1=t1,t2=t2,phi=phi,k=[kx,ky],M=M)
        Hx=get_H(t1=t1,t2=t2,phi=phi,k=[kx+0.001,ky],M=M)
        Hy=get_H(t1=t1,t2=t2,phi=phi,k=[kx,ky+0.001],M=M)
        H_dx = (Hx-H)/0.001# partial H/partial x
        H_dy = (Hy-H)/0.001# partial H/partial y

        #e,n 
        v,w= np.linalg.eig(H) # eigenvalue and eigenstate
        e=[]
        if v[0]>v[1]:
            e.extend([v[1],v[0]])
            n=w[::-1]
        else:
            e.extend([v[0],v[1]])
            n=w
        
        bc1= 1j*(np.dot(np.dot(n[0],H_dx),n[1].T)*np.dot(np.dot(n[1],H_dy),n[0].T)\
            -np.dot(np.dot(n[0],H_dy),n[1].T)*np.dot(np.dot(n[1],H_dx),n[0].T))/((e[0]-e[1])**2)
        bc2=1j*(np.dot(np.dot(n[1],H_dx),n[0].T)*np.dot(np.dot(n[0],H_dy),n[1].T)\
            -np.dot(np.dot(n[1],H_dy),n[0].T)*np.dot(np.dot(n[0],H_dx),n[1].T))/((e[1]-e[0])**2)
        
        #print(bc1,bc2)        
        # should be real number, drop numerical error 
        berry_curv_1.append(bc1.real)
        berry_curv_2.append(bc2.real)
        berry_curv_total.append(bc1.real+bc2.real)



    chern_number=get_berryphase(kxlist=kxlist,kylist=kylist,berry_curv=berry_curv_1) ##change which band to use!

    ## plot and save
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 5)
    fig.subplots_adjust(wspace=0.4)
    tick1=np.arange(-5,5.05,0.05)
    tick2=np.arange(-5,5.05,0.05)
    plt.yticks(tick2)
    plt.xticks(tick1)
    ax.axis('off')

    im=ax.tricontourf(kxlist,kylist,berry_curv_1)  ##change which band to use!
    ax.set_aspect(1)
    #cax = add_right_cax(ax, pad=0.01, width=0.02)
    cbar = fig.colorbar(im, shrink =0.6,format='%.2f',extend='max'\
        ,label='Berry curvature')
    temp="Chern num={:.6f} t1={} t2={} M={} phi={:.2f} band_1".format(chern_number,t1,t2,M,phi)##change which band to use!
    plt.title(temp)
    plt.savefig('./figs/'+temp+'.png',dpi=800)
    print("OK")

if __name__=='__main__':
    main(t1=1,t2=0.05,M=0.,phi=pi/2)
# %%
