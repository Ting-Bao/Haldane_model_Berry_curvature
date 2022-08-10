'''
Ting BAO @ Tsinghua University
2022/07/26

A training task on Haldane model: 
1. Calculate the Berry curvature distribution in the BZ
2. Calculate the Berry phase
3. Calculate the sgn(M)-depedent Chern number
'''


from math import cos, e, pi, sqrt
from xmlrpc.client import FastMarshaller
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

npoints=100
d_step=1e-10
klength=4*pi/sqrt(3)
step=12.05/npoints
# lattice vector

# NN: neareat neighbour, mark each element a_i 
# NNN: next nearest neighbour, mark each element b_i
NN = np.array([[0.5,sqrt(3)/6],[-0.5,sqrt(3)/6],[0.,-sqrt(3)/3]])
NNN = np.array([[-0.5,sqrt(3)/2],[-0.5,-sqrt(3)/2],[1.,0.]])

# !!!!!!!!!!!!!!!!!!!!
#  the d_size should be normalized!
# !!!!!!!!!!!!!!!!!!!!

dsize=(step**2)/(sqrt(3)/2*(klength**2))
V=sqrt(3)/2

#
# to be debug !
#

# this is wrong
dsize=step**2/(((2*pi)**2)/V)

# this is right
dsize=(step**2)/(np.linalg.norm(np.cross([-0.5,sqrt(3)/2,0],[-0.5,-sqrt(3)/2,0]),ord=2))
print(np.cross([-0.5,sqrt(3)/2,0],[-0.5,-sqrt(3)/2,0]))
print(np.linalg.norm(2*pi/V*np.cross([-0.5,sqrt(3)/2,0],[-0.5,-sqrt(3)/2,0])))


def get_H(t1,t2,phi,k,M):
    '''t1 for NN hopping, 
    t2 for NNN hopping,
    phi for phase in NNN hopping,
    return Hamiltonian matrix as k=[kx,ky] under given parameter
    '''

    H11=2*t2*(sum([cos(phi+np.dot(k,bi)) for bi in NNN]))+M
    H12=t1*(sum([e**(-1j*np.dot(k,ai)) for ai in NN]))
    H21=t1*(sum([e**(1j*np.dot(k,ai)) for ai in NN]))
    H22=2*t2*sum([cos(phi-np.dot(k,bi)) for bi in NNN])-M

    H=np.array([[H11,H12],[H21,H22]])
    return H


def get_berryphase(kxlist,kylist,berry_curv,dsize=dsize):
    """ intergrate in a rectangle area to get the Berry phase 
    dsize means the step using in meshing the BZ
    the length of recipical lattice vector is 4*pi/sqrt(3)
    """

    phase=0.
    for i in range(len(kxlist)):
        kx=kxlist[i]
        ky=kylist[i]
        if kx>=0 and kx<klength*sqrt(3)/2:
            if ky>=(1/sqrt(3))*kx and ky<(1/sqrt(3))*kx+klength:
                phase=phase+dsize*berry_curv[i]
    print(phase)
    return phase


def main(t1,t2,M,phi):
    '''parameters, change as required
    '''

    # do k-mesh
    sample_kx=np.linspace(0,12.05,npoints,endpoint=False)
    sample_ky=np.linspace(0,12.05,npoints,endpoint=False)
    xx,yy=np.meshgrid(sample_kx,sample_ky)
    kxlist=xx.flatten()
    kylist=yy.flatten()
    # each (kx[i],ky[i]) mark a k sample point in the BZ

    berry_curv_1=[]
    berry_curv_2=[]
    berry_curv_total=[]

    for i in tqdm(range(len(kxlist))):
        kx=kxlist[i]
        ky=kylist[i]
        H=get_H(t1=t1,t2=t2,phi=phi,k=[kx,ky],M=M)
        Hx=get_H(t1=t1,t2=t2,phi=phi,k=[kx+d_step,ky],M=M)
        Hy=get_H(t1=t1,t2=t2,phi=phi,k=[kx,ky+d_step],M=M)
        H_dx = (Hx-H)/d_step# partial H/partial x
        H_dy = (Hy-H)/d_step# partial H/partial y

        #print(H)
        #e,n 
        v,w= np.linalg.eig(H) # eigenvalue and eigenstate
        #!!!!!!!!
        #!!!!!!!!
        if v[0].real>v[1].real:
            e=[v[1],v[0]]
            n=[w[:,1],w[:,0]]
        else:
            e=[v[0],v[1]]
            n=[w[:,0],w[:,1]]

        bc1= np.dot(np.dot(n[0],H_dx),n[1])*np.dot(np.dot(n[1],H_dy),n[0])/((e[0].real-e[1].real)**2)
        bc2= np.dot(np.dot(n[1],H_dx),n[0])*np.dot(np.dot(n[0],H_dy),n[1])/((e[1].real-e[0].real)**2)
        
        # print(bc1,bc2)        
        # should be real number, drop numerical error 
        
        berry_curv_1.append(2*bc1.imag)
        berry_curv_2.append(2*bc2.imag)
        berry_curv_total.append(2*bc1.imag+2*bc2.imag)

    chern_number=get_berryphase(kxlist=kxlist,kylist=kylist,berry_curv=berry_curv_1)/2/pi ##change which band to use!
    print("chern_number = {}".format(chern_number))

    ## plot and save
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 5)
    fig.subplots_adjust(wspace=0.4)
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
    main(t1=1,t2=0.,M=0.5,phi=0)