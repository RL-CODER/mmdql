from copy import deepcopy
import numpy as np
from mushroom_rl.core import Agent
from mushroom_rl.approximators.regressor import Ensemble, Regressor
from replay_memory import ReplayMemory

class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.
    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size,
                 approximator_params, target_update_frequency,
                 fit_params=None, n_approximators=1, clip_reward=True):
        """
        Constructor.
        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            approximator_params (dict): parameters of the approximator to
                build;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            n_approximators (int, 1): the number of approximator to use in
                ``AverageDQN``;
            clip_reward (bool, True): whether to clip the reward or not.
        """
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train["name"] = "train"
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target["name"] = "target"
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)
        if self._n_approximators == 1:
            self.target_approximator.model.set_weights(
                self.approximator.model.get_weights())
        else:
            for i in range(self._n_approximators):
                self.target_approximator.model[i].set_weights(
                    self.approximator.model.get_weights())

        super(DQN, self).__init__(mdp_info, policy)

    def fit(self, dataset, lives = None, episode_frame_number = None, frame_number = None):
        mask = np.ones((len(dataset), self._n_approximators))
        self._replay_memory.add(dataset, mask)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, mask =\
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self.approximator.fit(state, action, q, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        """
        Update the target network.
        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.
        Returns:
            Maximum action-value for each state in ``next_state``.
        """
        q = self.target_approximator.predict(next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)

    def draw_action(self, state):
        action = super(DQN, self).draw_action(np.array(state))

        return action


class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al.. 2016.
    """
    def _next_q(self, next_state, absorbing):
        q = self.approximator.predict(next_state)
        max_a = np.argmax(q, axis=1)

        double_q = self.target_approximator.predict(next_state, max_a)
        if np.any(absorbing):
            double_q *= 1 - absorbing

        return double_q