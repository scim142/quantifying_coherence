import numpy as np
from scipy import optimize

def kl_loss(p, q): #the disimilarity function f, summed over all entries
    return (p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))).sum()


def loss_function(V, q, loss): #returns a function pi -> loss(p = V pi, q)
    return lambda pi: loss(np.matmul(V, pi), q)

def compute_loss(q, V, loss):
    N = V.shape[1]
    constraint= optimize.LinearConstraint(
        np.concatenate((np.eye(N), [np.ones(N)])),
        lb = (np.arange(N+1) == N) + 0, #first N are all atoms have non-negative probabiliites, last is that they must sum to 1
        ub = (np.ones(N+1)) #first N are redudant that atoms must have probabities <=1, last is they must sum to 1
    )
    res = optimize.minimize(
        fun = loss_function(V, q, loss),
        x0 = np.ones(N) / N, #initialize with even distribution across atoms
        constraints= (constraint)
        
    )

    return res


if __name__ == "__main__":
    V = np.array([[1, 0, 0], [1, 0, 0]])
    q = np.array([0.5, 0.6])

    print(compute_loss(q, V, kl_loss))