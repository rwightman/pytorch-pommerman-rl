import torch
import torch.nn as nn


class SIL:
    def __init__(
            self,
            algo,
            update_ratio=1.0,
            epochs=1,
            value_loss_coef=0.01,
            batch_size=64,
            entropy_coef=0.,

    ):
        self.update_ratio = update_ratio
        self.epochs = epochs
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.avg_loss_by_valid_samples = True

        self.algo = algo

    def _calc_num_updates(self, index):
        num_updates = 0
        if self.update_ratio < 1:
            denom = int(round(1 / self.update_ratio))
            if index % denom == 0:
                num_updates = 1
        else:
            num_updates = int(round(self.update_ratio))
        return num_updates

    def update(self, rollouts, index, replay=None):
        value_loss, action_loss, dist_entropy = self.algo.update(rollouts, index)

        num_updates = self._calc_num_updates(index)
        if replay is not None and replay.num_steps > self.batch_size and num_updates:
            sil_value_loss, sil_action_loss = self.update_sil(replay, num_updates, self.epochs)
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
                data_generator = replay.feed_forward_generator(self.batch_size, num_updates_per_epoch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, weights_batch, indices_batch = sample

                values, action_log_probs, dist_entropy, _ = self.algo.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                advantages = (return_batch - values)
                clipped_advantages = torch.clamp(advantages, min=0.0)

                # FIXME this loss is what's described in the paper, but the author's TF implementation differs.
                # TODO Look into the TF implementation, it appears to be motivated by the author's
                # lower-bound-soft-Q-learning equivalence justification.
                # https://github.com/junhyukoh/self-imitation-learning/blob/master/baselines/common/self_imitation.py

                action_loss = -action_log_probs * clipped_advantages.detach()
                value_loss = 0.5 * clipped_advantages.pow(2)

                # apply importance sampling (priority sampling bias correction) weights
                if weights_batch is not None:
                    action_loss *= weights_batch
                    value_loss *= weights_batch

                if self.avg_loss_by_valid_samples:
                    num_valid_samples = torch.clamp(torch.sum(advantages > 0).float(), min=1.0)
                    action_loss = action_loss.sum() / num_valid_samples
                    value_loss = value_loss.sum() / num_valid_samples
                else:
                    action_loss = action_loss.mean()
                    value_loss = value_loss.mean()

                loss = value_loss * self.value_loss_coef + action_loss
                if self.entropy_coef:
                    loss -= dist_entropy * self.entropy_coef

                self.algo.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.algo.actor_critic.parameters(), self.algo.max_grad_norm)

                self.algo.optimizer.step()

                replay.update_priorities(indices_batch, clipped_advantages)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                num_updates += 1

        if num_updates:
            value_loss_epoch /= num_updates
            action_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch

