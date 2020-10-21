import time

import numpy as np

from bandits import BernoulliBandit
from solvers import EpsilonGreedy, UCB1, BayesianUCB, ThompsonSampling, Exp3
from plotter import plot


def main():
    """
    Run a small experiment for a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.
    """

    np.random.seed(int(time.time()))

    # Number of slot machines
    K = 10
    # Number of time steps
    N = 5000

    b = BernoulliBandit(K)
    print('Randomly generated Bernoulli bandit has reward probabilities:')
    print(b.probas)
    print()
    print(f'The best machine has index: {b.best_i} and proba: {b.best_proba}')

    solvers = [
        EpsilonGreedy(r'$\epsilon$-Greedy', b, 0.01),
        UCB1('UCB1', b),
        BayesianUCB('Bayesian UCB', b, 3, 1, 1),
        ThompsonSampling('Thompson Sampling', b, 1, 1),
        Exp3('EXP3', b, 0.01, 1),
    ]

    for s in solvers:
        s.run(N)

    plot(solvers, f'results_K{K}_N{N}.png')


if __name__ == '__main__':
    main()
