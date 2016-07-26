import numpy as np
import scipy as sp
import scipy.sparse.linalg as ssl
import numpy.linalg as npl
from scipy.integrate import quad
from time import time
import matplotlib.pyplot as plt



def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper


def transverIsing():
    """
	Main function to run transver Ising model with the iTEBD algorithm for transverse Ising model
	Hamiltonian is: H= -J sum_i sigma_z,i sigma_z,i+1, G for Gamma, l for lambda
	"""
    # -----------------parameter setting----------------------
    J,g=1,5
    epislon=0.005 # delta beta
    chi,d=100,2
    G = np.random.rand(2,d,chi,chi) # G[A,:,:,:] and G[B,:,:,:]
    l = np.random.rand(2,chi) # l[A,:] and G[B,:]
    H=np.array([[J,-g/2,-g/2,0],[-g/2,-J,0,-g/2],[-g/2,0,-J,-g/2],[0,-g/2,-g/2,J]])
    t_series=np.linspace(1,4,50)
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    E0_exact = quad(f, 0, np.pi, args=(g,))[0]
    # energyVsT(t_series,H,epislon,G,l,E0_exact)
    theta,G,l=evolve(H,2,epislon,G,l)
    spectrum(l)
    print("E_iTEBD =", -np.log(np.sum(theta**2))/epislon/2)
    print("E_exact =", E0_exact)


def evolve(H,t_fin,epislon,G,l):
    [d,chi]=np.shape(G[0,:,:,0])
    N=int(t_fin/epislon)
    w,v=np.linalg.eig(H)
    U=np.reshape(np.dot(np.dot(v,np.diag(np.exp(-epislon*w))),np.transpose(v)),[2,2,2,2])
    for step in range(N):
        A=np.mod(step,2)
        B=np.mod(step+1,2)
        # Construct theta
        theta = np.tensordot(np.diag(l[B,:]),G[A,:,:,:],axes=(1,1))
        theta = np.tensordot(theta,np.diag(l[A,:],0),axes=(2,0))
        theta = np.tensordot(theta,G[B,:,:,:],axes=(2,1))
        theta = np.tensordot(theta,np.diag(l[B,:],0),axes=(3,0))
        # Apply imaginary-time evolve
        theta = np.tensordot(theta,U,axes=([1,2],[0,1]))
        theta=np.reshape(np.transpose(theta,(2,0,3,1)),[d*chi,d*chi])
        X,Y,Z=np.linalg.svd(theta)
        Z=Z.transpose()
        # Truncate the bond dimension back to chi and normalize the state
        l[A,0:chi]=Y[0:chi]/np.sqrt(sum(Y[0:chi]**2))
        X=np.reshape(X[0:d*chi,0:chi],(d,chi,chi))
        G[A,:,:,:]=np.transpose(np.tensordot(np.diag(l[B,:]**(-1)),X,axes=(1,1)),(1,0,2))
        Z=np.transpose(np.reshape(Z[0:d*chi,0:chi],(d,chi,chi)),(0,2,1))
        G[B,:,:,:]=np.tensordot(Z,np.diag(l[B,:]**(-1)),axes=(2,0))
    return theta,G,l

def energyVsT(t_series,H,epislon,G,l,E0_exact):
    erro=np.zeros(len(t_series))
    for (i,t) in enumerate(t_series):
        theta,G,l=evolve(H,t,epislon,G,l)
        erro[i]=abs(-np.log(np.sum(theta**2))/epislon/2-E0_exact)/abs(E0_exact)
    plt.figure()
    plt.plot(t_series,erro)
    plt.yscale("log")
    plt.show()


def spectrum(l):
    plt.figure("Spectrum")
    plt.plot(l[0,:]**2)
    plt.plot(l[1,:]**2)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()



def transferOP2(A,O):
    """
    Calculates the transfer matrix of the one-site operator O and the MPS A
    which is \prod_k sum_{ik,jk} <jk|Ok|ik> A[k]^ik kron A'[k]jk
    """
    [D,D,d]=np.shape(A)
    K=np.reshape(A,[D**2,d])
    K=np.dot(A,O)
    R=np.reshape(np.conj(A),[D**2,d])
    E=np.dot(K,np.transpose(R))
    E=np.reshape(E,[D,D,D,D])
    E=E.transpose([0,2,1,3])
    E=np.reshape(E,[D**2,D**2])
    return E


def mpoAppl(A,C):
    """
    Applies the iMPO with matrices C to the iMPS with matrices A

    """
    [D,D,d]=np.shape(A)
    Dc=np.shape(C)[0]
    L=np.reshape(A,[D*D,d])
    T=np.reshape(C,[Dc*Dc*d,d])
    K=np.dot(L,np.transpose(T))
    erg=np.reshape(K,[D,D,Dc,Dc,d])
    erg=np.transpose(erg,[2,0,3,1,4])
    erg=np.reshape(erg,[D*Dc,D*Dc,d])
    return erg


def normal(A):
    """
    Normalization of the translational invariant iMPS given by A
    It also calculates the eigenvector to the biggest eigenvalue of the transfer-operator of A.
    This vector is used in the function energyIs to save time.
    """
    d=np.shape(A)[2]
    E=transferOP2(A,np.eye(d))
    [e,v]=ssl.eigs(E,k=1,which='LM')
    erg=A/np.sqrt(e)
    return [erg,v]


transverIsing()







