import torch
import torch.nn as nn


class SIL:
    def __init__(
            self,
            algo,
            sil_update_ratio=1.0,
            sil_epochs=1,
            sil_value_loss_coef=0.01,
            sil_batch_size=64,
            sil_entropy_coef=0.01,

    ):
        self.sil_update_ratio = sil_update_ratio
        self.sil_epochs = sil_epochs
        self.sil_value_loss_coef = sil_value_loss_coef
        self.sil_batch_size = sil_batch_size
        self.sil_entropy_coef = sil_entropy_coef

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
            sil_value_loss, sil_action_loss = self.update_sil(replay, num_updates, self.sil_epochs)
            print("SIL: value_loss = {:.5f}, action_loss = {:.5f}".format(sil_value_loss, sil_action_loss))

        return value_loss, action_loss, dist_entropy

    def update_sil(self, replay, num_updates_per_epoch=1, num_epochs=1):
        value_loss_epoch = 0
        action_loss_epoch = 0
        num_updates = 0

        for _ in range(num_epochs):
            if self.algo.actor_critic.is_recurrent:
                assert False, "Not implemented"
            else:
                data_generator = replay.feed_forward_generator(self.sil_batch_size, num_updates_per_epoch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch = sample

                values, action_log_probs, dist_entropy, _ = self.algo.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                advantages = (return_batch - values)
                clipped_advantages = torch.clamp(advantages, min=0.0)

                #FIXME experiment with scaling by num_valid_samples after verifying the rest of the impl
                #num_valid_samples = torch.sum(advantages > 0).float()
                #action_loss = (-action_log_probs * clipped_advantages.detach()).sum() / num_valid_samples
                #value_loss = (clipped_advantages.pow(2)).sum() / (2 * num_valid_samples)

                #FIXME this loss is what's described in the paper, but the author's TF implementation differs.
                #TODO Look into the TF implementation, it appears to be motivated by the author's
                #lower-bound-soft-Q-learning equivalence justification.
                action_loss = -(action_log_probs * clipped_advantages.detach()).mean()
                value_loss = 0.5 * (clipped_advantages.pow(2)).mean()
                loss = value_loss * self.sil_value_loss_coef + action_loss
                if self.sil_entropy_coef:
                    loss -= dist_entropy * self.sil_entropy_coef

                self.algo.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.algo.actor_critic.parameters(), self.algo.max_grad_norm)

                self.algo.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                num_updates += 1

        if num_updates:
            value_loss_epoch /= num_updates
            action_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch

