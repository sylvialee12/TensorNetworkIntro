"""
This is the DMRG variational algorithm to find the ground state of an MPO.
"""

import numpy as np
import numpy.random as random
import scipy.sparse.linalg as ssl
import numpy.linalg as nl
import matplotlib.pyplot as plt
from time import time


def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper

class DMRG():
    """
    The DMRG variational algorithm to find the ground state of an MPO, here we take XXZ model as an example
    """
    def __init__(self,eps,MPO,N,D):
        """
        :param eps: The convergence criteria
        :param MPO: The MPO of a system
        :param N: the site number
        :param D: bond dimension
        """
        self.MPO=MPO
        self.d=np.shape(MPO[1])[0] # physical dimension
        self.Dw=np.shape(MPO[1])[2] # MPO bond dimension
        self.eps=eps
        self.N=N
        self.D=D


    def initialize(self):
        """
        Initialize the wave function
        :return: A random initial MPS
        """
        N=self.N
        M=[]
        a=random.rand(self.d,1,self.D)
        M.append(a)
        for i in range(1,N-1):
            a=random.rand(self.d,self.D,self.D)
            M.append(a)
        a=random.rand(self.d,self.D,1)
        M.append(a)
        return M


    def right_canonical(self,T):
        """
        :param T: a tensor on specific site
        :return: T:Right-canonical tensor,
        """
        d,D1,D2=np.shape(T)
        T2=np.transpose(T,[1,0,2])
        b=T2.reshape(D1,d*D2)
        B,U=nl.qr(np.transpose(np.conj(b)))
        B=np.transpose(np.conj(B))
        d1,d2=B.shape
        T=np.transpose(B.reshape((d1*d2)//(d*D2),d,D2),[1,0,2])
        return T,np.transpose(np.conj(U))


    def left_cannonical(self,T):
        """
        :param T: a tensor representing specific site
        :return: Left-canonical tensor
        """
        d,D1,D2=np.shape(T)
        T2=np.reshape(T,[D1*d,D2])
        A,U=nl.qr(T2)
        d1,d2=A.shape
        T=np.reshape(A,[d,D1,d1*d2//(D1*d)])
        return T,U


    def right_operator(self,M):
        """
        :param M: the initial MPS
        :return: the right-canonical MPS B and the right operator
        """
        # B=[self.right_canonical(m)[0] for m in M]
        right_cannonical=self.right_canonical
        B=[[] for i in range(len(M))]
        B[-1],U=right_cannonical(M[-1])
        for i in range(1,len(M)-1):
            T=np.tensordot(M[-(i+1)],U,axes=(2,0))
            B[-(i+1)],U=right_cannonical(T)
        B[0]=np.tensordot(M[0],U,axes=(2,0))
        R=[[] for i in range(len(M))]
        Ri = np.tensordot(B[-1], self.MPO[-1], axes=(0, 0))
        Ri = np.tensordot(Ri, np.conj(B[-1]), axes=(2, 0))
        R[-1] = np.transpose(Ri, [0, 2, 4, 1, 3, 5])
        R[-1]=np.squeeze(R[-1])
        for i in range(1,len(M)):
            Ri = np.tensordot(B[-(i+1)],R[-i],axes=(2,0))
            Ri = np.tensordot(self.MPO[-(i+1)], Ri,axes=([0,3],[0,2]))
            Ri = np.tensordot(np.conj(B[-(i+1)]),Ri,axes=([0,2],[0,3]))
            R[-(i+1)] = np.transpose(Ri,[2,1,0])
        return R,B




    def right_sweep(self,R,M):
        """
        :param R: The right operator
        :param M:
        :return:
        """
        H=np.tensordot(self.MPO[0],R[1],axes=(3,1))
        H=np.squeeze(H)
        H=np.transpose(H,[0,2,1,3])
        d0,d1,d2,d3=H.shape
        H=np.reshape(H,[d0*d1,d2*d3])
        w,v=ssl.eigsh(H,which='SA',k=1,maxiter=5000)
        v=np.reshape(v,[self.d,1,d0*d1//self.d])
        l,u=self.left_cannonical(v)
        M[0]=l
        L=[[] for i in range(len(R))]
        L[0]=np.tensordot(l,self.MPO[0],axes=(0,0))
        L[0]=np.tensordot(L[0],np.conj(l),axes=(2,0))
        L[0]=np.transpose(L[0],[0,2,4,1,3,5])
        for i in range(1,len(R)-1):
            H=np.tensordot(self.MPO[i],R[i+1],axes=(3,1))
            H=np.tensordot(L[i-1],H,axes=(4,2))
            H=H.squeeze()
            H=np.transpose(H,[2,0,4,3,1,5])
            d1,d2,d3,d4,d5,d6=H.shape
            H=np.reshape(H,[d1*d2*d3,d1*d2*d3])
            w,v=ssl.eigsh(H,which='SA',k=1,maxiter=5000)
            v = np.reshape(v, [d1, d2, d3])
            l, u = self.left_cannonical(v)
            M[i] = l
            Li = np.tensordot(L[i-1],l,axes=(3,1))
            Li = np.tensordot(Li, self.MPO[i],axes=([5,3],[0,2]))
            L[i] = np.tensordot(Li, np.conj(l),axes=([3,5],[1,0]))
        M[-1]=np.tensordot(u,M[-1],axes=(1,1))
        return L,M

    def left_sweep(self,L,M):
        H = np.tensordot(L[-2], self.MPO[-1], axes=(4, 2))
        H = np.squeeze(H)
        H = np.transpose(H, [2,0,3,1])
        d0,d1,d2,d3=H.shape
        H = np.reshape(H, [d0*d1, d2 * d3])
        w, v = ssl.eigsh(H, which='SA', k=1,maxiter=5000)
        v=np.reshape(v,[self.d,d0*d1//self.d,1])
        v,u=self.right_canonical(v)
        M[-1] = v
        R=[[] for i in range(len(L))]
        R[-1]=np.tensordot(v,self.MPO[-1],axes=(0,0))
        R[-1]=np.tensordot(R[-1],np.conj(v),axes=[2,0])
        R[-1]=np.transpose(R[-1],[0,2,4,1,3,5])
        for i in range(1,len(L)-1):
            H = np.tensordot(L[-(i+2)],self.MPO[-(i+1)], axes=(4, 2))
            H = np.tensordot(H,R[-i], axes=(7, 1))
            H = np.squeeze(H)
            H = np.transpose(H, [2,0, 4, 3, 1, 5])
            d0,d1,d2,d3,d4,d5=H.shape
            H = np.reshape(H, [d0*d1*d2, d3*d4*d5])
            w, v = ssl.eigsh(H, which='SA', k=1,maxiter=5000)
            v=np.reshape(v,[d0,d1,d2])
            v,u=self.right_canonical(v)
            M[-(i+1)] = v
            Ri = np.tensordot(np.conj(v),R[-i],axes=(2,2))
            Ri = np.tensordot(self.MPO[-(i+1)],Ri,axes=([1,3],[0,3]))
            R[-(i+1)] = np.tensordot(v, Ri,axes=([0,2],[0,3]))
        M[0]=np.tensordot(M[0],u,axes=(2,0))
        return R,M

    def site_expecation_value(self, M, O,i):

        if i==0:
            e0=np.tensordot(M[0],O,axes=(0,0))
            e=np.tensordot(e0,np.conj(M[0]),axes=([0,1,2],[1,2,0]))
        else:
            ei=np.tensordot(M[i],O,axes=(0,0))
            ei=np.tensordot(ei,np.conj(M[i]),axes=([2,1],[0,2]))
            m0=np.tensordot(M[0],np.conj(M[0]),axes=([0,1],[0,1]))
            for j in range(1,i):
                mi=np.tensordot(M[j],np.conj(M[j]),axes=(0,0))
                m0=np.tensordot(m0,mi,axes=([0,1],[0,2]))
            e=np.tensordot(m0,ei,axes=([0,1],[0,1]))

        norm=np.tensordot(M[0],np.conj(M[0]),axes=([0,1,2],[0,1,2]))

        return np.real(e/norm)


    def energy(self,R1,M0):
        R0=np.tensordot(M0,self.MPO[0],axes=(0,0))
        R0=np.tensordot(R0,np.conj(M0),axes=(2,0))
        R0=np.transpose(R0,[0,2,4,1,3,5])
        energy1=np.squeeze(np.tensordot(R0,R1,axes=([3,4,5],[0,1,2])))
        norm=np.tensordot(M0,np.conj(M0),axes=([0,1,2],[0,1,2]))
        return energy1/norm


    def norm(self,M0):
        norm=np.tensordot(M0,np.conj(M0),axes=([0,1,2],[0,1,2]))
        return norm

    # Not tested yet
    def correlation(self,M,operator,site_i,site_j):
        """
        Calculation for the correlation between site_i and site_j <|OiOj|>-<Oi><Oj>
        :param operator: operator

        """
        minsite = min(site_i,site_j)
        maxsite = max(site_i,site_j)
        u = np.array([[1]])
        for i in range(0,minsite):
            M[i] = np.tensordot(u, M[i],axes=(-1,1)).transpose(1,0,2)
            l,u = self.left_cannonical(M[i])
            M[i] = l
        M[minsite] = np.tensordot(u, M[minsite]).transpose(1,0,2)
        MP = np.tensordot(M[minsite],operator,axes=(0,0))
        MPI = np.tensordot(MP, np.conj(M[minsite]),axes=(-1,0))
        MPI = MPI.transpose([0,2,1,3])
        for i in range(minsite+1,maxsite):
            MI = np.tensordot(MPI, M[i],axes=(2,1))
            MPI = np.tensordot(MI, np.conj(M[i]), axes=([3,2],[0,1]))
            # MI = np.tensordot(M[i], np.conj(M[i]), axes=(0,0)).transpose(0,2,1,3)
            # MPI = np.tensordot(MPI, MI,axes=([2,3],[0,1]))

        MP = np.tensordot(M[maxsite],operator,axes=(0,0))
        MPJ = np.tensordot(MP, np.conj(M[maxsite]),axes=(-1,0))
        MPJ = MPJ.transpose([0,2,1,3])

        product = np.tensordot(MPI,MPJ, axes=([2,3,0,1]))
        correlation = np.trace(product)

        return correlation



    def magnetization(self,M):
        sz = np.array([[1, 0], [0, -1]])
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        mz=np.zeros(self.N)
        mx=np.zeros(self.N)
        my=np.zeros(self.N)
        site_expectation=self.site_expecation_value
        for i in range(self.N):
            mz[i]=site_expectation(M,sz,i)
            mx[i]=site_expectation(M,sx,i)
            my[i]=site_expectation(M,sy,i)
        return np.mean(np.abs(mx)),np.mean(my),np.mean(np.abs(mz))



    def meanmagnetization(self):
        pass



    @functimer
    def dmrg(self):
        M00 = self.initialize()
        R,M = self.right_operator(M00)
        L,M0 = self.right_sweep(R,M)
        R,M1 = self.left_sweep(L,M0)
        n=0
        s=0
        for (m, m1) in zip(M, M1):
            s = max(abs(m - m1).max(), s)
        while s>self.eps and n<100:
            print(s)
            M=M1
            L,M0=self.right_sweep(R,M)
            R,M1=self.left_sweep(L,M0)
            n+=1
            for (m, m1) in zip(M, M1):
                s = max(abs(m - m1).max(), s)
        return M,R

def mpo(N,h,J,Jz):
    sz = np.array([[1,0],[0,-1]])/2
    sp = np.array([[0,1],[0,0]])
    sm = np.array([[0,0],[1,0]])
    MPO=[[] for i in range(N)]
    MPO[1] = np.zeros([2,2,5,5])
    MPO[1][:,:,0,0] = np.eye(2)
    MPO[1][:,:,1,0] = sp
    MPO[1][:,:,2,0] = sm
    MPO[1][:,:,3,0] = sz
    MPO[1][:,:,4,0] = -h*(sm+sp)/2
    MPO[1][:,:,4,1] = J/2*sm
    MPO[1][:,:,4,2] = J/2*sp
    MPO[1][:,:,4,3] = Jz*sz
    MPO[1][:,:,4,4] = np.eye(2)
    MPO[0]=MPO[1][:,:,4,:].reshape(2,2,1,5)
    MPO[N-1]=MPO[1][:,:,:,0].reshape(2,2,5,1)
    for i in range(2,N-1):
        MPO[i]=MPO[1]
    return MPO



if __name__=="__main__":
    N = [10,20,30]
    J,Jz = 0,-1
    H = np.linspace(0,1,20)
    mx = np.zeros(20)
    my = np.zeros(20)
    mz = np.zeros(20)
    for n in N:
        for (idx,h) in enumerate(H):
            MPO = mpo(n,h,J,Jz)
            a=DMRG(10**(-7),MPO,n,20)
            M,R=a.dmrg()
            energy=a.energy(R[1],M[0])/n
            print(energy)
            mx[idx], my[idx] ,mz[idx] = a.magnetization(M)

        plt.figure("Magnetization")
        plt.plot(H,mz)
    plt.show()





