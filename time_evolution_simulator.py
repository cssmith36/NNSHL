import jax
import jax.numpy as jnp
import numpy as np
from operations import ket0, pX, pZ
from jax.scipy.linalg import expm


def timeEvolution(psi,H,t):
    return expm(-1.j*H*t)@psi

#Hamil = 1.0*(jnp.kron(pX,jnp.eye(2)) + 
#             jnp.kron(jnp.eye(2),pX)) + \
#             0.1* jnp.kron(pZ,pZ)

#tEvol = timeEvolution(jnp.kron(ket0,ket0),Hamil,1.)



#print(Hamil)
#print(jnp.outer(tEvol,tEvol.conj()))
#print(jnp.trace(jnp.outer(tEvol,tEvol.conj())))

