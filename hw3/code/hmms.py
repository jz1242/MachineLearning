import numpy as np


def forward(x, pi, A, B):
    """ Run the forward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        alpha, a 2-D float NumPy array with shape [T, N_z].
    """
    t = x.shape[0]
    n_z = pi.shape[0]
    alpha = np.zeros((t, n_z))
    alpha[0, :] = B[:, x[0]]*pi[:]
    for i in range(1, t):
        alpha[i, :] = B[:, x[i]] * (A[0, :]*alpha[i-1, 0] + A[1, :]*alpha[i-1, 1])
    return alpha
    
def backward(x, pi, A, B):
    """ Run the backward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        beta, a 2-D float NumPy array with shape [T, N_z].
    """
    t = x.shape[0]
    n_z = pi.shape[0]
    beta = np.zeros((t, n_z))
    beta[t-1, :] = 1
    for i in range(t - 2, -1, -1):
        beta[i, :] = (A[:, 0]*B[0, x[i + 1]]*beta[i+1,0]) + (A[:, 1]*B[1, x[i + 1]]*beta[i+1,1])
    return beta

def individually_most_likely_states(X, pi, A, B):
    """ Computes individually most-likely states.

    By "individually most-likely states," we mean that the *marginal*
    distributions are maximized. In other words, for any particular
    time step of any particular sequence, each returned state i is
    chosen to maximize P(z_t = i | x).

    All sequences in X are assumed to have the same length, T.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        Z, a 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., N_z - 1.
    """
    # TODO: Write this function.
    n = X.shape[0]
    t = X.shape[1]
    z = np.zeros((n, t), dtype=int)
    for i in range(0, X.shape[0]):
        alpha = forward(X[i], pi, A, B)
        beta = backward(X[i], pi, A, B)
        p_x = np.sum(alpha[t - 1])
        p_zx = (1/p_x)*alpha*beta
        z[i] = np.argmax(p_zx, axis=1)
    return z


def take_EM_step(X, pi, A, B):
    """ Take a single expectation-maximization step.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        A tuple containing
        pi_prime: pi after the EM update.
        A_prime: A after the EM update.
        B_prime: B after the EM update.
    """
    n_z = pi.shape[0]
    t = X.shape[1]
    n_x = B.shape[1]
    pi_prime = np.zeros(n_z, dtype=float)
    A_prime = np.zeros((n_z, n_z))
    B_prime = np.zeros((n_z, n_x))
    for i in range(0, X.shape[0]):
        x = X[i]
        alpha = forward(X[i], pi, A, B)
        beta = backward(X[i], pi, A, B)
        p_x = np.sum(alpha[t - 1])
        pi_prime += (1/p_x)*alpha[0,:]*beta[0,:]
        for j in range(0, 2):
            for k in range(0, 2):
                A_prime[j,k] += (1/p_x)*A[j,k]*np.dot((np.multiply(alpha[:-1, j], B[k, x[1:]] ),),beta[1:, k])
                B_prime[j,k] += (1/p_x)*(np.dot(alpha[x == k, j], beta[x == k, j]))

    tot = np.sum(pi_prime)
    for m in range(0, n_z):
        pi_prime[m] /= tot
    for n in range(0, n_z):
        sum = np.sum(A_prime[n])
        for l in range(0, n_z):
            A_prime[n, l] = A_prime[n, l]/sum
    for q in range(0, n_z):
        sum = np.sum(B_prime[q])
        for s in range(0, n_x):
            B_prime[q, s] = B_prime[q, l]/sum

    return pi_prime, A_prime, B_prime

    
