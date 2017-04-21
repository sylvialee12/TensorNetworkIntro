import numpy as np
import scipy as sp
import scipy.sparse.linalg as ssl
import numpy.linalg as npl
from scipy.integrate import quad
from time import time
import matplotlib.pyplot as plt



sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])
sy=np.array([[0,-1j],[1j,0]])


def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper

@functimer
def transverIsing():
    """
	Main function to run transver Ising model with the iTEBD algorithm for transverse Ising model
	Hamiltonian is: H= -J sum_i sigma_z,i sigma_z,i+1, G for Gamma, l for lambda
	"""
    # -----------------parameter setting----------------------
    J,g=1,1
    epislon=0.005 # delta beta
    chi,d=10,2
    t_fin=5
    G = np.random.rand(2,d,chi,chi) # G[A,:,:,:] and G[B,:,:,:]
    l = np.random.rand(2,chi) # l[A,:] and G[B,:]
    H=np.array([[J,-g/2,-g/2,0],[-g/2,-J,0,-g/2],[-g/2,0,-J,-g/2],[0,-g/2,-g/2,J]])
    t_series=np.linspace(1,4,50)
    g_series=np.linspace(0,5,50)
    # mx,my,mz=magVsg(g_series,J,t_fin,epislon,G,l)
    # plotmag(mx,my,mz,g_series)
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    E0_exact = quad(f, 0, np.pi, args=(g,))[0]
    energyVsT(t_series,H,epislon,G,l,E0_exact)
    theta,G,l = evolve(H,5,epislon,G,l)
    mz = site_expecation_value(G,l,sz)
    mx = site_expecation_value(G,l,sx)
    energy = bond_expectation_value(G,l,H)
    print("E_iTEBD =", -np.log(np.sum(theta**2))/epislon/2)
    print("E_exact =", E0_exact)
    print(mz)
    print(mx)
    print(energy)



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


def magVsg(g_series,J,t_fin,epislon,G,l):
    mx,my,mz=np.zeros([2,len(g_series)]),np.zeros([2,len(g_series)]),np.zeros([2,len(g_series)])
    for (i,g) in enumerate(g_series):
        H = np.array([[J, -g / 2, -g / 2, 0], [-g / 2, -J, 0, -g / 2], [-g / 2, 0, -J, -g / 2], [0, -g / 2, -g / 2, J]])
        theta,G,l=evolve(H,t_fin,epislon,G,l)
        mx[:,i]=site_expecation_value(G,l,sx)
        # my[:,i]=site_expecation_value(G,l,sy)
        mz[:,i]=site_expecation_value(G,l,sz)
    return mx,my,mz


def plotmag(mx,my,mz,g):
    plt.figure("Magnetization")
    plt.subplot(1,3,1)
    plt.plot(g,mx[0,:])
    plt.subplot(1,3,2)
    plt.plot(g,my[0,:])
    plt.subplot(1,3,3)
    plt.plot(g,mz[0,:])
    plt.tight_layout()
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

def entanglement_entropy(l):
    S=np.zeros(2)
    for i_bond in range(2):
        x=l[i_bond,:]**2
        S[i_bond]=np.dot(-np.log(x),x)
    return S

def site_expecation_value(G,l,O):
    E=np.zeros(2)
    for isite in range(0,2):
        A=isite%2
        B=(isite+1)%2
        theta=np.tensordot(np.diag(l[B,:]),G[A,:,:,:],axes=(1,1))
        theta=np.tensordot(theta,np.diag(l[A,:]),axes=(2,0))
        theta_o=np.tensordot(theta,O,axes=(1,0)).conj()
        E[isite]=np.squeeze(np.tensordot(theta_o,theta,axes=([0,1,2],[0,2,1]))).item()
    return E


def bond_expectation_value(G,l,U):
    E=np.zeros(2)
    for isite in range(2):
        A=isite%2
        B=(isite+1)%2
        theta = np.tensordot(np.diag(l[B, :]), G[A, :, :, :], axes=(1, 1))
        theta = np.tensordot(theta, np.diag(l[A, :]), axes=(2, 0))
        theta = np.tensordot(theta,G[B,:,:,:],axes=(2,1))
        theta = np.tensordot(theta,np.diag(l[B,:]),axes=(3,0))
        theta_o = np.tensordot(theta, np.reshape(U,[2,2,2,2]), axes=([1,2],[0,1])).conj()
        E[isite] = np.squeeze(np.tensordot(theta_o, theta, axes=([0, 2,3,1], [0, 1,2,3]))).item()
    return E


def correlation(G,l,n,O):
    corr=np.zeors(2)
    for isite in range(0,2):
        for i in range(n):
            theta=np.tensordot(np.diag(l[(isite+i+1)%2,:]),G[(isite+i)%2,:,:,:],axes=(1,1))
            theta=np.tensordot(theta,np.diag(l[(isite+i)%2,:]),axes=(2,0))
        theta_o=np.tensordot(theta,O,axes=(1,0))
        theta_o=np.tensordot(theta_o,O,axes=(-2,0)).conj()
        oder_o=[0]+list(range(2,n+1))+[n+2]+[1,n+1]
        corr[isite]=np.squeeze(np.tensordot(theta_o,theta,axes=(list(range(n+3)),oder_o))).item()
    return corr


def Heisenberg():

    sp = np.array([[0,1],[0,0]])
    sm = np.array([[0,0],[1,0]])
    sz = np.array([[1,0],[0,-1]])/2
    s0 = np.eye(2)
    hz, Jxx, Jz = 0, 1, 1
    epislon=0.005 # delta beta
    chi,d=10,2

    H = Jxx/2*(np.kron(sp,sm)+np.kron(sm,sp))+Jz*(np.kron(sz,sz))+\
        hz*(np.kron(sz,s0))+hz*(np.kron(s0,sz))

    G = np.random.rand(2,d,chi,chi) # G[A,:,:,:] and G[B,:,:,:]
    l = np.random.rand(2,chi) # l[A,:] and G[B,:]

    theta,G,l = evolve(H,20,epislon,G,l)
    mz = site_expecation_value(G,l,sz)
    mx = site_expecation_value(G,l,sx)
    energy = bond_expectation_value(G,l,H)
    print("E_iTEBD =", -np.log(np.sum(theta**2))/epislon/2)

    print(energy)


Heisenberg()





