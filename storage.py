import torch
import numpy as np
from collections import deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = torch.zeros(num_steps, num_processes, 1).long()
        else:
            self.actions = torch.zeros(num_steps, num_processes, action_space.shape[0])
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class ReplayStorage:
    def __init__(self, max_steps, num_processes, gamma, obs_shape, action_space, recurrent_hidden_state_size):
        self.max_steps = max_steps
        self.num_processes = num_processes
        self.gamma = gamma

        # stored episode data
        self.obs = torch.zeros(max_steps, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(max_steps, recurrent_hidden_state_size)
        self.returns = torch.zeros(max_steps, 1)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = torch.zeros(max_steps, 1).long()
        else:
            self.actions = torch.zeros(max_steps, action_space.shape[0])
        self.masks = torch.ones(max_steps, 1)
        self.next_idx = 0
        self.num_steps = 0

        # store (full) episode stats
        self.episode_step_count = 0
        self.episode_rewards = deque()
        self.episode_steps = deque()

        # currently running (accumulating) episodes
        self.running_episodes = [[] for _ in range(num_processes)]

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

    def insert(self, obs, rhs, actions, rewards, dones):
        for n in range(self.num_processes):
            self.running_episodes[n].append(dict(
                obs=obs[n],
                rhs=rhs[n],
                action=actions[n],
                reward=rewards[n]
            ))
        for n, done in enumerate(dones):
            if done:
                self._add_trajectory(self.running_episodes[n])
                self.running_episodes[n] = []

    def feed_forward_generator(self, batch_size, num_batches=None):
        batch_count = 0
        indices = np.random.permutation(range(self.num_steps))
        for si in range(0, len(indices), batch_size):
            batch_indices = indices[si:max(len(indices), si + batch_size)]
            if len(batch_indices) < batch_size:
                return

            obs_batch = self.obs[batch_indices].cuda()
            recurrent_hidden_states_batch = self.recurrent_hidden_states[batch_indices].cuda()
            actions_batch = self.actions[batch_indices].cuda()
            returns_batch = self.returns[batch_indices].cuda()
            masks_batch = self.masks[batch_indices].cuda()
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, returns_batch, masks_batch

            batch_count += 1
            if num_batches and batch_count >= num_batches:
                return
