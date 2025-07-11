from copy import deepcopy
import numpy as np
from mushroom_rl.core import Agent
from mushroom_rl.approximators.regressor import Ensemble, Regressor
from replay_memory import  ReplayMemory

class BootstrappedDQN(Agent):
    def __init__(self, approximator, policy, mdp_info, batch_size, target_update_frequency, initial_replay_size, max_replay_size, fit_params=None, approximator_params=None, 
                    n_approximators=1, clip_reward=True, p_mask=2 / 3.):
        self._fit_params = dict() if fit_params is None else fit_params
        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency
        self._p_mask = p_mask
        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)
        self._n_updates = 0
        self._episode_steps = 0
        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train['name'] = 'train'
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target['name'] = 'target'
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator, **apprx_params_target)
        policy.set_q(self.approximator)
        self.target_approximator.model.set_weights(self.approximator.model.get_weights())
        super(BootstrappedDQN, self).__init__(mdp_info, policy)

    def fit(self, dataset, lives=None, episode_frame_number=None, frame_number=None):
        mask = np.random.binomial(1, self._p_mask, size=(len(dataset), self._n_approximators))
        self._replay_memory.add(dataset, mask)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, mask = self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward.reshape(self._batch_size, 1) + self.mdp_info.gamma * q_next
            self.approximator.fit(state, action, q, mask=mask, **self._fit_params)
            self._n_updates += 1
            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        self.target_approximator.model.set_weights(self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        q = np.array(self.target_approximator.predict(next_state))[0]
        for i in range(q.shape[1]):
            if absorbing[i]:
                q[:, i, :] *= 1. - absorbing[i]
        max_q = np.max(q, axis=2)

        return max_q.T

    def draw_action(self, state):
        action = super(BootstrappedDQN, self).draw_action(np.array(state))
        self._episode_steps += 1

        return action

    def episode_start(self):
        self._episode_steps = 0
        self.policy.set_idx(np.random.randint(self._n_approximators))