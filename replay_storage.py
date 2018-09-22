import torch
import numpy as np
import math
import random
from collections import deque
from helpers.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayStorage:
    def __init__(
            self, max_steps, num_processes, gamma, prio_alpha,
            obs_shape, action_space, recurrent_hidden_state_size,
            device):
        self.max_steps = int(max_steps)
        self.num_processes = num_processes
        self.gamma = gamma
        self.device = device

        # stored episode data
        self.obs = torch.zeros(self.max_steps, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(self.max_steps, recurrent_hidden_state_size)
        self.returns = torch.zeros(self.max_steps, 1)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = torch.zeros(self.max_steps, 1).long()
        else:
            self.actions = torch.zeros(self.max_steps, action_space.shape[0])
        self.masks = torch.ones(self.max_steps, 1)
        self.next_idx = 0
        self.num_steps = 0

        # store (full) episode stats
        self.episode_step_count = 0
        self.episode_rewards = deque()
        self.episode_steps = deque()

        # currently running (accumulating) episodes
        self.running_episodes = [[] for _ in range(num_processes)]

        if prio_alpha > 0:
            """
            Sampling priority is enabled if prio_alpha > 0
            Priority algorithm ripped from OpenAI Baselines
            https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
            """
            self.prio_alpha = prio_alpha
            tree_capacity = 1 << math.ceil(math.log2(self.max_steps))
            self.prio_sum_tree = SumSegmentTree(tree_capacity)
            self.prio_min_tree = MinSegmentTree(tree_capacity)
            self.prio_max = 1.0
        else:
            self.prio_alpha = 0

    def _process_rewards(self, trajectory):
        has_positive = False
        reward_sum = 0.
        r = 0.
        for t in trajectory[::-1]:
            reward = t['reward']
            reward_sum += reward
            if reward > (0. + 1e-5):
                has_positive = True
            r = reward + self.gamma*r
            t['return'] = r
        return has_positive, reward_sum

    def _add_trajectory(self, trajectory):
        has_positive, reward_sum = self._process_rewards(trajectory)
        if not has_positive:
            return
        trajectory_len = len(trajectory)
        prev_idx = self.next_idx
        for transition in trajectory:
            self.obs[self.next_idx].copy_(transition['obs'])
            self.recurrent_hidden_states[self.next_idx].copy_(transition['rhs'])
            self.actions[self.next_idx].copy_(transition['action'])
            self.returns[self.next_idx].copy_(transition['return'])
            self.masks[self.next_idx] = 1.0
            prev_idx = self.next_idx
            if self.prio_alpha:
                self.prio_sum_tree[self.next_idx] = self.prio_max ** self.prio_alpha
                self.prio_min_tree[self.next_idx] = self.prio_max ** self.prio_alpha
            self.next_idx = (self.next_idx + 1) % self.max_steps
            self.num_steps = min(self.max_steps, self.num_steps + 1)
        self.masks[prev_idx] = 0.0

        # update stats of stored full trajectories (episodes)
        while self.episode_step_count + trajectory_len > self.max_steps:
            steps_popped = self.episode_steps.popleft()
            self.episode_rewards.popleft()
            self.episode_step_count -= steps_popped
        self.episode_step_count += trajectory_len
        self.episode_steps.append(trajectory_len)
        self.episode_rewards.append(reward_sum)

    def _sample_proportional(self, sample_size):
        res = []
        for _ in range(sample_size):
            mass = random.random() * self.prio_sum_tree.sum(0, self.num_steps - 1)
            idx = self.prio_sum_tree.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def insert(self, obs, rhs, actions, rewards, dones):
        for n in range(self.num_processes):
            self.running_episodes[n].append(dict(
                obs=obs[n].clone(),
                rhs=rhs[n].clone(),
                action=actions[n].clone(),
                reward=rewards[n].clone()
            ))
        for n, done in enumerate(dones):
            if done:
                self._add_trajectory(self.running_episodes[n])
                self.running_episodes[n] = []

    def update_priorities(self, indices, priorities):
        if not self.prio_alpha:
            return

        """Update priorities of sampled transitions.
        sets priority of transition at index indices[i] in buffer
        to priorities[i].
        Parameters
        ----------
        indices: [int]
            List of indices of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled indices.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < self.num_steps
            self.prio_sum_tree[idx] = priority ** self.prio_alpha
            self.prio_min_tree[idx] = priority ** self.prio_alpha

            self.prio_max = max(self.prio_max, priority)

    def feed_forward_generator(self, batch_size, num_batches=None, beta=0.):
        """Generate batches of sampled experiences.

        Parameters
        ----------
        batch_size: int
            Size of each sampled batch
        num_batches: int
            Number of batches to sample
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """

        batch_count = 0
        sample_size = num_batches * batch_size or self.num_steps

        if self.prio_alpha > 0:
            indices = self._sample_proportional(sample_size)
            if beta > 0:
                # compute importance sampling weights to correct for the
                # bias introduced by sampling in a non-uniform manner
                weights = []
                p_min = self.prio_min_tree.min() / self.prio_sum_tree.sum()
                max_weight = (p_min * self.num_steps) ** (-beta)
                for i in indices:
                    p_sample = self.prio_sum_tree[i] / self.prio_sum_tree.sum()
                    weight = (p_sample * self.num_steps) ** (-beta)
                    weights.append(weight / max_weight)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            else:
                weights = torch.ones((len(indices), 1), dtype=torch.float32)
        else:
            if sample_size * 3 < self.num_steps:
                indices = random.sample(range(self.num_steps), sample_size)
            else:
                indices = np.random.permutation(self.num_steps)[:sample_size]
            weights = None

        for si in range(0, len(indices), batch_size):
            indices_batch = indices[si:min(len(indices), si + batch_size)]
            if len(indices_batch) < batch_size:
                return

            weights_batch = None if weights is None else \
                weights[si:min(len(indices), si + batch_size)].to(self.device)

            obs_batch = self.obs[indices_batch].to(self.device)
            recurrent_hidden_states_batch = self.recurrent_hidden_states[indices_batch].to(self.device)
            actions_batch = self.actions[indices_batch].to(self.device)
            returns_batch = self.returns[indices_batch].to(self.device)
            masks_batch = self.masks[indices_batch].to(self.device)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, returns_batch, \
                  masks_batch, weights_batch, indices_batch

            batch_count += 1
            if num_batches and batch_count >= num_batches:
                return
