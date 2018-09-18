import copy
import glob
import os
import time
import types
from collections import deque

import numpy as np
import torch

import algo
from arguments import get_args
from envs import make_vec_envs
from models import create_policy
from storage import RolloutStorage, ReplayStorage
from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

update_factor = args.num_steps * args.num_processes
num_updates = int(args.num_frames) // update_factor
lr_update_schedule = None if args.lr_schedule is None else args.lr_schedule // update_factor

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"
try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    train_envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma, args.no_norm, args.num_stack,
        args.log_dir, args.add_timestep, device, allow_early_resets=False)

    if args.eval_interval:
        eval_envs = make_vec_envs(
            args.env_name, args.seed + args.num_processes, args.num_processes, args.gamma,
            args.no_norm, args.num_stack, eval_log_dir, args.add_timestep, device,
            allow_early_resets=True, eval=True)

        if eval_envs.venv.__class__.__name__ == "VecNormalize":
            eval_envs.venv.ob_rms = train_envs.venv.ob_rms
    else:
        eval_envs = None

    actor_critic = create_policy(
        train_envs.observation_space,
        train_envs.action_space,
        name='pomm',
        nn_kwargs={
            'batch_norm': False if args.algo == 'acktr' else True,
            'image_shape': [9, 11, 11],
            'recurrent': args.recurrent_policy,
            'hidden_size': 512,
        })

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr, lr_schedule=lr_update_schedule,
            eps=args.eps, alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
        agent = algo.SIL(agent)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
            args.value_loss_coef, args.entropy_coef,
            lr=args.lr, lr_schedule=lr_update_schedule,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef,
            args.entropy_coef,
            acktr=True)

    if True:
        replay = ReplayStorage(
            5000,
            args.num_processes,
            args.gamma,
            train_envs.observation_space.shape,
            train_envs.action_space,
            actor_critic.recurrent_hidden_state_size)
    else:
        replay = None

    rollouts = RolloutStorage(
        args.num_steps, args.num_processes,
        train_envs.observation_space.shape,
        train_envs.action_space,
        actor_critic.recurrent_hidden_state_size)

    obs = train_envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = train_envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
            if replay is not None:
                replay.insert(obs, recurrent_hidden_states, action, reward, done)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, j, replay)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model.state_dict(),
                            hasattr(train_envs.venv, 'ob_rms') and train_envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * update_factor

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {}, last {} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if args.eval_interval and len(episode_rewards) > 1 and j > 0 and j % args.eval_interval == 0:
            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 50:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass


if __name__ == "__main__":
    main()
