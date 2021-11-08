#Given W

import numpy as np
h11 = 50
n = float(h11)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

#standard variannce
var = 1.0/np.sqrt(h11)

#geometric data
k1 = np.random.normal(loc = 0, scale = var,size = h11)
k2p = np.random.normal(loc = 0, scale = var,size = [h11,h11])
k2 = k2p + np.transpose(k2p)
k3p = np.random.normal(loc = 0, scale = var,size = [h11,h11,h11])
k3 = 1/6.0*(k3p + np.einsum('ikj',k3p) + np.einsum('jik',k3p) + np.einsum('jki',k3p) + np.einsum('kij',k3p) + np.einsum('kji',k3p))


#k3 =  np.full((h11,h11,h11),1)



#R4 = np.random.normal(loc = 0, scale = var,size = [h11,h11,h11,h11])
#R4 = np.random.normal(loc = 0, scale = var,size = [h11,h11,h11,h11])
k4p = np.random.normal(loc = 0, scale = var,size = [h11,h11,h11,h11])
k4 =  1/24.0*(k4p + np.einsum('ikjl',k4p) + np.einsum('jikl',k4p) + np.einsum('jkil',k4p) + np.einsum('kijl',k4p) + np.einsum('kjil',k4p))
+ 1/24.0*(k4p + np.einsum('iklj',k4p) + np.einsum('jilk',k4p) + np.einsum('jkli',k4p) + np.einsum('kilj',k4p) + np.einsum('kjli',k4p))
+ 1/24.0*(k4p + np.einsum('ilkj',k4p) + np.einsum('jlik',k4p) + np.einsum('jlki',k4p) + np.einsum('klij',k4p) + np.einsum('klji',k4p))
+ 1/24.0*(k4p + np.einsum('likj',k4p) + np.einsum('ljik',k4p) + np.einsum('ljki',k4p) + np.einsum('lkij',k4p) + np.einsum('lkji',k4p))

R4p = np.random.normal(loc = 0, scale = var,size = [h11,h11,h11,h11])
R4 =  1/24.0*(R4p + np.einsum('ikjl',R4p) + np.einsum('jikl',R4p) + np.einsum('jkil',R4p) + np.einsum('kijl',R4p) + np.einsum('kjil',R4p))
+ 1/24.0*(R4p + np.einsum('iklj',R4p) + np.einsum('jilk',R4p) + np.einsum('jkli',R4p) + np.einsum('kilj',R4p) + np.einsum('kjli',R4p))
+ 1/24.0*(R4p + np.einsum('ilkj',R4p) + np.einsum('jlik',R4p) + np.einsum('jlki',R4p) + np.einsum('klij',R4p) + np.einsum('klji',R4p))
+ 1/24.0*(k4p + np.einsum('likj',R4p) + np.einsum('ljik',R4p) + np.einsum('ljki',R4p) + np.einsum('lkij',R4p) + np.einsum('lkji',R4p))

#superpotential and its derivatives, drawn randomly
W = np.random.normal(loc = 0, scale = var)
W1 = np.random.normal(loc = 0, scale = var,size = h11)
pW2 = np.random.normal(loc = 0, scale = var,size = [h11,h11])
pW3 = np.random.normal(loc = 0, scale = var,size = [h11,h11,h11])

#symmetrize:
W2 = 0.5*(pW2 + np.transpose(pW2))
W3 = 1/6.0*np.array([[[pW3[a][b][c] + pW3[a][c][b] + pW3[b][a][c]+ pW3[b][c][a] + pW3[c][a][b] + pW3[c][b][a] for a in range(h11)]
	for b in range(h11)] for c in range(h11)])

#metric inverse:
k2inv = np.linalg.inv(k2)

#christoffels, last index is up:
Gamma = np.einsum('ijk,kl',k3,k2inv)

#F-terms:
F1 = W1 + W*k1

# #F-term derivative
# dF1 = W2 + W*k2 + np.einsum('i,j',W1,k1)

#Z-matrix:
Z2 = W2 + W*k2 + np.einsum('i,j',k1,W1) + np.einsum('i,j',W1,k1) + W*np.einsum('i,j',k1,k1) - np.einsum('ijk,k',Gamma,F1)

#U-tensor
#fully symmetrized version:

U1 = W3 + k3*W + W*np.einsum('i,j,k',k1,k1,k1) - np.einsum('lm,mijk,l',k2inv,k4,F1)
U2 = np.einsum('ij,k',k2,W1) + np.einsum('ik,j',k2,W1) + np.einsum('jk,i',k2,W1)
U3 = np.einsum('ij,k',W2,k1) + np.einsum('ik,j',W2,k1) + np.einsum('jk,i',W2,k1)
U4 = np.einsum('ij,k',k2,k1*W) + np.einsum('ik,j',k2,k1*W) + np.einsum('jk,i',k2,k1*W)
U5 = np.einsum('i,j,k',k1,k1,W1) + np.einsum('k,j,i',k1,k1,W1) + np.einsum('i,k,j',k1,k1,W1)
U6 = -np.einsum('jil,kl',Gamma,W2) -np.einsum('jkl,il',Gamma,W2) -np.einsum('ikl,jl',Gamma,W2)
U7 = -np.einsum('jil,kl',Gamma,k2*W) -np.einsum('jkl,il',Gamma,k2*W) -np.einsum('ikl,jl',Gamma,k2*W)
U8 = -np.einsum('jil,k,l',Gamma,k1,k1*W) -np.einsum('jkl,i,l',Gamma,k1,k1*W) -np.einsum('kil,j,l',Gamma,k1,k1*W)
U9 = -np.einsum('jil,k,l',Gamma,W1,k1) -np.einsum('jkl,i,l',Gamma,W1,k1) -np.einsum('kil,j,l',Gamma,W1,k1)
U10 = -np.einsum('jil,k,l',Gamma,k1,W1) -np.einsum('jkl,i,l',Gamma,k1,W1) -np.einsum('kil,j,l',Gamma,k1,W1)
U11 = np.einsum('klm,ijl,m',Gamma,Gamma,W1) + np.einsum('jlm,ikl,m',Gamma,Gamma,W1) + np.einsum('ilm,kjl,m',Gamma,Gamma,W1)
U12 = np.einsum('klm,ijl,m',Gamma,Gamma,k1*W) + np.einsum('jlm,ikl,m',Gamma,Gamma,k1*W) + np.einsum('ilm,kjl,m',Gamma,Gamma,k1*W)

U = U1 + U2 + U3 + U4 + U5 + U6 + U7 + U8 + U9 + U10 + U11 + U12
#print U
#print U[0,0,1], U[0,1,0], U[1,0,0]

Habarb = np.einsum('il,lk,jk',Z2,k2inv,np.conj(Z2)) - np.einsum('i,j',F1,np.conj(F1)) - np.einsum('ijkl,km,ln,m,n',R4,k2inv,k2inv,np.conj(F1),F1)
Hbarab = np.conj(Habarb)
Hab = np.einsum('ijk,kl,l',U,k2inv,np.conj(F1)) - Z2*np.conj(W)
Hbarabarb = np.conj(Hab)
H = np.block([[Habarb, Hab],[Hbarabarb, Hbarab]]) 
+ np.block([[k2, np.full((h11,h11),0)],[np.full((h11,h11),0), k2]])*(np.einsum('i,ik,k',F1,k2inv,F1) - 2*W*np.conj(W))
print np.linalg.eig(H)
print check_symmetric(H)





