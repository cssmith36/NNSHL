import numpy as np
import jax
import jax.numpy as jnp
from operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, hadamard, phase_z, pX, pZ
from time_evolution_simulator import timeEvolution

num_qubits = 2
densities = [[denseXP,denseXM],[denseYP,denseYM],[denseZP,denseZM]]
unitaries = jnp.array([hadamard, hadamard @ phase_z, jnp.array([[1.,0.],[0.,1.]])])

def constructProbs(rho):
    paulis = [0,1,2]
    probabilities = []
    probDict = {}    
    order = []
    for i in range(3):
        for ii in range(3):
            tmpP = []
            order.append([paulis[i],paulis[(i+ii)%3]])
            for j in range(2):
                for k in range(2):
                    prob = np.trace(rho@np.kron(densities[i][j],densities[(i+ii)%3][k]))
                    tmpP.append(np.real(prob))
            probabilities.append(tmpP)
    order = np.array(order)
    for i in range(len(order)):
        probDict[str(order[i])] = probabilities[i]
    return probDict

def classicalShadowCalc(rho,shadow_size,num_qubits):
    
    ### Construct the list of random Paulis to measure (can feed it a predefined seed for consistency/reproducibility)
    #np.random.seed(seed)
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    
    outcomes = []
    probDict = constructProbs(rho)
    
    ### Randomly generating the outcomes of the measurements
    for ns in range(shadow_size):
        rand = np.random.uniform(0,1)
        if rand <= probDict[str(unitary_ids[ns])][0]:
            outcomes.append([1,1])
        elif probDict[str(unitary_ids[ns])][0] < rand <= probDict[str(unitary_ids[ns])][0] + probDict[str(unitary_ids[ns])][1]:
            outcomes.append([1,-1])
        elif probDict[str(unitary_ids[ns])][0] + probDict[str(unitary_ids[ns])][1] < rand <= probDict[str(unitary_ids[ns])][0] + probDict[str(unitary_ids[ns])][1] + probDict[str(unitary_ids[ns])][2]:
            outcomes.append([-1,1])
        else:
            outcomes.append([-1,-1])
    #print('outcomes',outcomes)
    #print('unitary ids:',unitary_ids)
    return (np.array(outcomes), unitary_ids)

def estimate_shadow_obervable(shadow, observable, k=10):
    """
    Adapted from https://github.com/momohuang/predicting-quantum-properties
    Calculate the estimator E[O] = median(Tr{rho_{(k)} O}) where rho_(k)) is set of k
    snapshots in the shadow. Use median of means to ameliorate the effects of outliers.

    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.
        observable (qml.Observable): Single PennyLane observable consisting of single Pauli
            operators e.g. qml.PauliX(0) @ qml.PauliY(1).
        k (int): number of splits in the median of means estimator.

    Returns:
        Scalar corresponding to the estimate of the observable.
    """
    shadow_size, num_qubits = shadow[0].shape

    target_obs, target_locs = observable
    #print(target_obs,target_locs)

    # classical values
    b_lists, obs_lists = shadow
    means = []

    #print('targetobs',target_obs)
    #print('targetloc',target_locs)
    # loop over the splits of the shadow:
    for i in range(0, shadow_size, shadow_size // k):

        # assign the splits temporarily
        b_lists_k, obs_lists_k = (
            b_lists[i: i + shadow_size // k],
            obs_lists[i: i + shadow_size // k],
        )

        # find the exact matches for the observable of interest at the specified locations
        indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)
        #print('indices',sum(indices))
        # catch the edge case where there is no match in the chunk
        if sum(indices) > 0:
            # take the product and sum
            product = np.prod(b_lists_k[indices][:, target_locs], axis=1)
            means.append(np.sum(product) / sum(indices))
        else:
            means.append(0)

    return np.median(means)



#estimate_shadow_obervable(rhoShadow, obs, k=1)

#rhoShadow = classicalShadowCalc(rho_t,20000,num_qubits)