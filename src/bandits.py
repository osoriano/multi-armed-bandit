from numpy.random import random


class BernoulliBandit:
    def __init__(self, n, probas=None):
        self.n = n
        self.probas = probas or [random() for _ in range(n)]
        self.best_proba = max(self.probas)
        self.best_i = self.probas.index(self.best_proba)

    def generate_reward(self, i):
        """Get reward for ith machine"""
        return random() < self.probas[i]
