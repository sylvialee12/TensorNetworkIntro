import numpy as np
import matplotlib.pyplot as plt
import scipy. sparse.linalg as ssl
import numpy.linalg as npl
import scipy.linalg as spl
from scipy.integrate import quad
from time import time
from collections import namedtuple


def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper


def mpsChain():
    Jxy,Jz,gx,gy,gz=0,1,0.5,0,0
    Para=namedtuple('Parameters',['I','t','ss','Dmpo','D','nsite'])
    para=Para(1,0.02,2,4,4,10)
    Pauli=namedtuple('Pauli',['S0','Sx','Sy','Sz'])
    pauli=Pauli(np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]]))
    H=-gx*np.kron(pauli.S0,pauli.Sx)-gx*np.kron(pauli.Sx,pauli.S0)-gy*np.kron(pauli.S0,pauli.Sy)-gy*np.kron(pauli.Sy,pauli.S0)-\
	  gz*np.kron(pauli.S0,pauli.Sz)-Jz*np.kron(pauli.Sz,pauli.Sz)-Jxy*np.kron(pauli.Sx,pauli.Sx)-gx*np.kron(pauli.Sy,pauli.Sy)




def initialization(para):
    d=para.ss
    D=para.D
    A=[]
    for i in range(para.nsite):
        A.append(np.random.rand(D,D,d))
    return A


def mpo(H,para):
    Ubond=spl.expm(-para.I*para.t*H)
    A=np.reshape(np.transpose(np.reshape(Ubond,[para.ss,para.ss,para.ss,para.ss]),[0,2,1,3]),[para.ss**2,para.ss**2])
    [u,s,v]=npl.svd(A,fullmatrix=False)
    u=np.dot(u,np.sqrt(s))
    v=np.dot(np.sqrt(s),np.transpose(np.conj(v)))
    mpobegin=np.zeros([para.ss,para.ss,para.Dmpo,para.Dmpo])
    mpoend=np.zeros([para.ss,para.ss,para.Dmpo,para.Dmpo])
    mpobegin[:,:,0,:]=np.transpose(np.reshape(u,[para.ss,para.ss,para.ss,para.Dmpo,1]),[0,1,3,2])
    mpoend[:,:,:,0]=np.transpose(np.reshape(v,[para.Dmpo,para.ss,para.ss,para.ss,1]),[1,2,0,3])

    A=np.reshape(np.transpose(np.reshape(u,[para.ss,para.ss,para.Dmpo]),[0,2,1]),[para.ss*para.Dmpo,para.ss])
    B=np.reshape(np.transpose(np.reshape(v,[para.Dmpo,para.ss,para.ss]),[1,0,2]),[para.ss,para.ss*para.Dmpo])
    mpoeven=np.transpose(np.reshape(np.dot(A,B),[para.Dmpo,para.ss,para.ss,para.Dmpo]),[1,2,0,3])

    A=np.reshape(v,[para.ss*para.Dmpo,para.ss])
    B=np.reshape(u,[para.ss,para.ss*para.Dmpo])
    mpoodd=np.transpose(np.reshape(np.dot(A,B),[para.Dmpo,para.ss,para.ss,para.Dmpo]),[1,2,0,3])
    mpo=[]
    mpo.append(mpobegin)
    for i in range(1,para.nsite):
        if (i-1)%2==0:
            mpo.append(mpoeven)
        else:
            mpo.append(mpoodd)
    mpo.append(mpoend)
    return mpo


def mpoL(H):
    pass


def mpoR(H):
    pass


def sweep(A):
    pass


def expectation(A,O):
    pass


def transfer(A,O):
    pass


def measure(para,pauli,T):
    Cleft=[np.zeros(para.D)]*(para.nsite+1)
    for i in range(para.nsite):
        Cleft[i+1]=UpdateCleft(para,Cleft[i],T[i],T[i])
    mag2=np.real(Cleft[para.nsite][0,0])
    return mag2

def UpdateCleft(para,Cleft,TA,TB):
    A=Contract(TA,Cleft,[2],[1])
    Cleft1=Contract(A,np.conj(TB),[1,3],[1,2])
    return Cleft1

def Contract():
    pass


