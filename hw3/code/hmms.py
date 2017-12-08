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
    for i in range(t - 2, 0, -1):
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
    #forward(X[0], pi, A, B)
    backward(X[0], pi, A, B)
    pass


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
    # TODO: Write this function.
    raise NotImplementedError('Not yet implemented.')
