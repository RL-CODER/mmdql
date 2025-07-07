import numpy as np
import tensorflow as tf
from mushroom_rl.policy.td_policy import TDPolicy
from mushroom_rl.utils.parameters import Parameter
from scipy.stats import norm
import traceback

class EpsGreedy(TDPolicy):
    def __init__(self, epsilon):
        super().__init__()

        assert isinstance(epsilon, Parameter)
        self._epsilon = epsilon

    def __call__(self, *args):
        state = args[0]
        q = self._approximator.predict(np.expand_dims(state, axis=0)).ravel()
        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilon.get_value(state) / self._approximator.n_actions

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilon.get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = np.ones(self._approximator.n_actions) * p
            probs[max_a] += (1. - self._epsilon.get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(state)
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def update(self, *idx):
        self._epsilon.update(*idx)

class BootPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(BootPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False
        self._idx = None
        self.plotter = None

    def set_plotter(self,plotter):
        self.plotter = plotter

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                max_as, count = np.unique(np.argmax(q_list, axis=1),
                                          return_counts=True)
                max_a = np.array([max_as[np.random.choice(
                    np.argwhere(count == np.max(count)).ravel())]])
                if self.plotter is not None:
                    self.plotter(np.array(q_list))
                return max_a
            else:
                q = self._approximator.predict(state, idx=self._idx)
                
                max_a = np.argwhere(q == np.max(q)).ravel()
                if len(max_a) > 1:
                    max_a = np.array([np.random.choice(max_a)])
                if self.plotter is not None:
                    self.plotter(np.array(self._approximator.predict(state)))
                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        self._idx = idx

    def update_epsilon(self, state):
        self._epsilon(state)


class WeightedPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(WeightedPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False
        self.plotter = None

    def set_plotter(self,plotter):
        self.plotter = plotter

    @staticmethod
    def _compute_prob_max(q_list):
        q_array = np.array(q_list).T
        score = (q_array[:, :, None, None] >= q_array).astype(int)
        prob = score.sum(axis=3).prod(axis=2).sum(axis=1)
        prob = prob.astype(np.float32)
        return prob / np.sum(prob)

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                means = np.mean(q_list, axis=0)
                max_a = np.array([np.random.choice(np.argwhere(means == np.max(means)).ravel())])
                if self.plotter is not None:
                    self.plotter(np.array(q_list))
                return max_a
            else:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for i in range(self._n_approximators):
                        q_list.append(self._approximator.predict(state, idx=i))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                qs = np.array(q_list)

                samples = np.ones(self._approximator.n_actions)
                for a in range(self._approximator.n_actions):
                    idx = np.random.randint(self._n_approximators)
                    samples[a] = qs[idx, a]

                max_a = np.array([np.random.choice(np.argwhere(samples == np.max(samples)).ravel())])
                if self.plotter is not None:
                    self.plotter(qs)
                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])
            
    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass

    def update_epsilon(self, state):
        self._epsilon(state)


class UCBPolicy(TDPolicy):
    def __init__(self, quantile_func=None, mu=None, delta=0.1, q_max=100):

        super(UCBPolicy, self).__init__()
        if quantile_func is None:
            self.quantile_func = lambda _: 0
        else:
            self.quantile_func = quantile_func
        if mu is None:
            self.mu =lambda _: 0
        else:
            self.mu = mu
        self.delta = delta
        self. q_max = q_max
        self._evaluation = False
        self.plotter = None

    def set_plotter(self,plotter):
        self.plotter = plotter

    def draw_action(self, state):
        means = self.mu(state)
        if self._evaluation:
            if self.plotter is not None:
                self.plotter(np.array(self._approximator.predict(state)))
            return np.array([np.random.choice(np.argwhere(means == np.max(means)).ravel())])
        bounds = self.quantile_func(state)
        a = np.array([np.random.choice(np.argwhere(bounds == np.max(bounds)).ravel())])
        if self.plotter is not None:
            self.plotter(np.array(self._approximator.predict(state)))
        return a

    def set_epsilon(self, epsilon):
        pass

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass

    def update_epsilon(self, state):
        pass


    def set_quantile_func(self, quantile_func):
        self.quantile_func = quantile_func

    def set_mu(self, mu):
        self.mu = mu