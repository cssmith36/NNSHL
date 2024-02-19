import jax.numpy as jnp



class Operations:

    def id(self):
        return jnp.eye(2)

    def pauliX(self):
        return jnp.array([[0.,1.],[1.,0.]])
    
    def pauliY(self):
        return jnp.array([[0.,-1.j],[1.j,0.]])
    
    def pauliZ(self):
        return jnp.array([[1.,0.],[0.,-1.]])

    def kron(self,x,y):
        kronProd = jnp.kron(x,y)
        return kronProd, kronProd

    def pauliString(self,string):
        init = string[0]
        return jax.lax.scan(self.kron,init,string[1:])[0]
    
    def basisState(self,bitstring):
        init = bitstring[0]
        return jax.lax.scan(self.kron,init,string[1:])[0]

ket0 = jnp.array([1,0]) # |0>
ket1 = jnp.array([0,1]) # |1>

ketXP = 1/jnp.sqrt(2)*jnp.array([1,1])  #  |+>
ketXM = 1/jnp.sqrt(2)*jnp.array([1,-1]) #  |->

ketYP = 1/jnp.sqrt(2)*jnp.array([1,1j]) #  |+Y>
ketYM = 1/jnp.sqrt(2)*jnp.array([1,-1j])#  |-Y>

H = 1/jnp.sqrt(2)*jnp.array([[1,1],[1,-1]])
HPhase = H@jnp.array([[1,0],[0,-1.j]])

denseZP = jnp.outer(ket0,ket0)
denseZM = jnp.outer(ket1,ket1)

denseXP = jnp.outer(ketXP,ketXP)
denseXM = jnp.outer(ketXM,ketXM)

denseYP = jnp.outer(ketYP,ketYP.conj())
denseYM = jnp.outer(ketYM,ketYM.conj())

zero_state = jnp.array([[1.,0.],[0.,0.]])
one_state = jnp.array([[0.,0.],[0.,1.]])

phase_z = jnp.array([[1, 0], [0, -1j]], dtype=complex)
hadamard = 1/jnp.sqrt(2) * jnp.array([[1.,1.],[1.,-1.]])

pX = jnp.array([[0.,1.],[1.,0.]])
pY = 1.j*jnp.array([[0.,-1.],[1.,0.]])
pZ = jnp.array([[1.,0.],[0.,-1.]])