import numpy as np
import random
from scipy.stats import beta


class Solver:
    def __init__(self, name, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        self.name = name
        self.bandit = bandit

        self.counts = [0 for _ in range(self.bandit.n)]
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0  # Cumulative regret.
        self.regrets = [0]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):
    def __init__(self, name, bandit, eps, init_proba=1):
        """
        eps: the probability (0-1) to explore at each time step
        init_proba: initial estimate of proba. Default is optimistic
        """
        super(EpsilonGreedy, self).__init__(name, bandit)

        assert 0 <= eps <= 1
        self.eps = eps

        self.estimates = [init_proba for _ in range(self.bandit.n)]

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one
            i = np.argmax(self.estimates)

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1 / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class UCB1(Solver):
    def __init__(self, name, bandit, init_proba=1):
        super(UCB1, self).__init__(name, bandit)
        self.t = 0
        self.estimates = [init_proba for _ in range(self.bandit.n)]

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            key=lambda x: self.estimates[x]
            + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x])),
        )
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1 / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, name, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BayesianUCB, self).__init__(name, bandit)
        self.c = c
        self._as = [init_a for _ in range(self.bandit.n)]
        self._bs = [init_b for _ in range(self.bandit.n)]

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x])
            + beta.std(self._as[x], self._bs[x]) * self.c,
        )
        r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += r
        self._bs[i] += 1 - r

        return i


class ThompsonSampling(Solver):
    def __init__(self, name, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(name, bandit)

        self._as = [init_a for _ in range(self.bandit.n)]
        self._bs = [init_b for _ in range(self.bandit.n)]

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [
            np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)
        ]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += 1 - r

        return i


class Exp3(Solver):
    def __init__(self, name, bandit, gamma, init_weight=1):
        super(Exp3, self).__init__(name, bandit)
        self.gamma = gamma
        self.weights = [init_weight for _ in range(self.bandit.n)]

    @property
    def estimated_probas(self):
        weights_sum = sum(self.weights)
        probas = [
            (1 - self.gamma) * w / weights_sum + self.gamma / self.bandit.n
            for w in self.weights
        ]
        return probas

    def run_one_step(self):
        weights_sum = sum(self.weights)
        probas = [
            (1 - self.gamma) * w / weights_sum + self.gamma / self.bandit.n
            for w in self.weights
        ]
        i = random.choices(range(self.bandit.n), probas)[0]

        r = self.bandit.generate_reward(i)
        self.weights[i] *= np.exp(self.gamma * r / probas[i] / self.bandit.n)

        return i
