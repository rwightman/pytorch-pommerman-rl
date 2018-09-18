import torch
import torch.nn as nn


class SIL:
    def __init__(self, algo, sil_update_ratio=1.0):
        self.sil_value_loss_coef = 0.01
        self.sil_update_ratio = 1.0
        self.sil_batch_size = 64
        self.algo = algo

    def _calc_num_updates(self, index):
        num_updates = 0
        if self.sil_update_ratio < 1:
            denom = int(round(1 / self.sil_update_ratio))
            if index % denom == 0:
                num_updates = 1
        else:
            num_updates = int(round(self.sil_update_ratio))
        return num_updates

    def update(self, rollouts, index, replay=None):
        value_loss, action_loss, dist_entropy = self.algo.update(rollouts, index)

        num_updates = self._calc_num_updates(index)
        if replay is not None and replay.num_steps > self.sil_batch_size and num_updates:
            sil_value_loss, sil_action_loss = self.update_sil(replay, num_updates)
            print("SIL: value_loss = {:.5f}, action_loss = {:.5f}".format(sil_value_loss, sil_action_loss))

        return value_loss, action_loss, dist_entropy

    def update_sil(self, replay, num_updates=1):
        value_loss_epoch = 0
        action_loss_epoch = 0
        num_batches = 0

        if self.algo.actor_critic.is_recurrent:
            assert False, "Not implemented"
        else:
            data_generator = replay.feed_forward_generator(self.sil_batch_size, num_updates)

        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, actions_batch, \
            return_batch, masks_batch = sample

            values, action_log_probs, dist_entropy, _ = self.algo.actor_critic.evaluate_actions(
                obs_batch, recurrent_hidden_states_batch,
                masks_batch, actions_batch)

            advantages = (return_batch - values)
            num_valid_samples = torch.sum(advantages > 0).float()
            clipped_advantages = torch.clamp(advantages, min=0.0)

            action_loss = (-action_log_probs * clipped_advantages.detach()).sum() / num_valid_samples
            value_loss = (clipped_advantages.pow(2)).sum() / (2 * num_valid_samples)

            self.algo.optimizer.zero_grad()

            (value_loss * self.sil_value_loss_coef + action_loss).backward()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()

            nn.utils.clip_grad_norm_(
                self.algo.actor_critic.parameters(), self.algo.max_grad_norm)

            self.algo.optimizer.step()

            num_batches += 1

        if num_batches:
            value_loss_epoch /= num_batches
            action_loss_epoch /= num_batches

        return value_loss_epoch, action_loss_epoch

