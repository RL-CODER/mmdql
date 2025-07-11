from copy import deepcopy
import numpy as np
import numexpr as ne
from mushroom_rl.core.agent import Agent
from mushroom_rl.approximators.regressor import Ensemble, Regressor
from replay_memory import ReplayMemory

class ParticleMMD_DQN(Agent):
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 target_update_frequency, initial_replay_size,
                 max_replay_size, fit_params=None, approximator_params=None,
                 n_approximators=1, clip_reward=True,
                 weighted_update=False, update_type='weighted', delta=0.1,
                 q_max=100, store_prob=False, max_spread=None):
        self._fit_params = dict() if fit_params is None else fit_params
        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency
        self.weighted_update = weighted_update
        self.update_type = update_type
        self.q_max = q_max
        self.store_prob = store_prob
        self.max_spread = max_spread
        quantiles = [i * 1. / (n_approximators - 1) for i in range(n_approximators)]
        for p in range(n_approximators):
            if quantiles[p] >= 1 - delta:
                self.delta_index = p
                break

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train['name'] = 'train'
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target['name'] = 'target'
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super(ParticleMMD_DQN, self).__init__(mdp_info, policy)
    
    @staticmethod
    def _compute_prob_max(q_list):
        q_array = np.array(q_list).T
        score = (q_array[:, :, None, None] >= q_array).astype(int)
        prob = score.sum(axis=3).prod(axis=2).sum(axis=1)
        prob = prob.astype(np.float32)
        return prob / np.sum(prob)

    @staticmethod
    def scale(x, out_range=(-1, 1), axis=None):
        domain = np.min(x, axis), np.max(x, axis)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

    def kernel_value(self, X, Y, inv_sigma=0.001, kernel_type='rbf'):
        if kernel_type == 'rbf':
            K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                    'A' : np.dot(X, X.T),
                    'B' : np.dot(Y, Y.T),
                    'C' : np.dot(X, Y.T),
                    'g' : inv_sigma
            })
            return K
        else:
            print("Unknown kernel type!")
            exit(1)

    def MMD_estimation(self, Q, V):
        return 1 + 1 - 2*self.kernel_value(Q, V)

    def fit(self, dataset, lives = None, episode_frame_number = None, frame_number = None):
        mask = np.ones((len(dataset), self._n_approximators))
        self._replay_memory.add(dataset, mask)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, mask =\
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next_pre, prob_explore = self._next_q(next_state, absorbing)
            q_next = np.zeros(q_next_pre.shape)    
            for head in range(q_next.shape[1]):
                exp_MMD_distance_list = []
                for head_2 in range(q_next.shape[1]):
                    if head != head_2:            
                        MMD_distance_list = []
                        for transition in range(q_next.shape[0]):
                            MMD_distance_list.append(self.MMD_estimation(q_next_pre[transition, head], q_next_pre[transition, head_2]))
                    else:
                        MMD_distance_list = []
                        for transition in range(q_next.shape[0]):
                            MMD_distance_list.append(99999999)
                    exp_MMD_distance_list.append(np.mean(np.array(MMD_distance_list)))
                q_next[:, head] = q_next_pre[:, exp_MMD_distance_list.index(min(exp_MMD_distance_list))]

            if self.max_spread is not None:
                for i in range(q_next.shape[0]):
                    min_range = np.min(q_next[i])
                    max_range = np.max(q_next[i])
                    if max_range - min_range > self.max_spread:
                        clip_range = (max_range - min_range) - self.max_spread
                        out_range = [min_range + clip_range / 2, max_range - clip_range / 2]
                        q_next[i] = ParticleMMD_DQN.scale(q_next[i], out_range=out_range, axis=None)
            q = reward.reshape(self._batch_size, 1) + self.mdp_info.gamma * q_next
            margin = 0.05
            self.approximator.fit(state, action, q, mask=mask,
                                  prob_exploration=prob_explore,
                                  margin=margin,
                                  **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        self.target_approximator.model.set_weights(self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        q = np.array(self.target_approximator.predict(next_state))[0]
        for i in range(q.shape[1]):
            if absorbing[i]:

                q[:, i, :] *= 0

        max_q = np.zeros((q.shape[1], q.shape[0]))
        prob_explore = np.zeros(q.shape[1])
        
        if self.update_type == 'mean':
            best_actions = np.argmax(np.mean(q, axis=0), axis=1)
            for i in range(q.shape[1]):
                max_q[i, :] = q[:, i, best_actions[i]]
                if self.store_prob:
                    particles = q[:, i, :]
                    particles = np.sort(particles, axis=0)
                    prob = ParticleMMD_DQN._compute_prob_max(particles)
                    prob_explore[i] = (1 - np.max(prob))
        elif self.update_type == 'weighted':
            for i in range(q.shape[1]):
                particles = q[:, i, :]
                particles = np.sort(particles, axis=0)
                prob = ParticleMMD_DQN._compute_prob_max(particles)
                max_q[i, :] = np.dot(particles, prob)
                if self.store_prob:
                    prob_explore[i] = (1 - np.max(prob))
        elif self.update_type == 'optimistic':
            for i in range(q.shape[1]):
                particles = q[:, i, :]
                particles = np.sort(particles, axis=0)
                means = np.mean(particles, axis=0)
                bounds = means + particles[self.delta_index, :]
                bounds = np.clip(bounds, -self.q_max, self.q_max)
                if self.store_prob:

                    prob = ParticleMMD_DQN._compute_prob_max(particles)
                    prob_explore[i] = (1 - np.max(prob))
                next_index = np.random.choice(np.argwhere(bounds == np.max(bounds)).ravel())
                max_q[i, :] = particles[:, next_index]

        else:
            raise ValueError("Update type not supported")

        if self.store_prob:
            return max_q, np.mean(prob_explore)

        return max_q, 0

    def draw_action(self, state):
        action = super(ParticleMMD_DQN, self).draw_action(np.array(state))

        return action

    def episode_start(self):
        self.policy.set_idx(np.random.randint(self._n_approximators))