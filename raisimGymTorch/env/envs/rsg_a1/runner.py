from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_a1 import RaisimGymEnv
from raisimGymTorch.env.bin.rsg_a1 import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
from collections import deque


if __name__ == '__main__':
    # task specification
    task_name = 'anymal_locomotion'

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default=None)
    args = parser.parse_args()
    weight_path = args.weight

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + '/../../../..'

    # config
    cfg = YAML().load(open(task_path + '/cfg.yaml', 'r'))

    # create environment from the configuration file
    env = VecEnv(RaisimGymEnv(home_path + '/rsc', dump(cfg['environment'], Dumper=RoundTripDumper)))
    obs = env.reset()

    # shortcuts
    ob_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    num_threads = cfg['environment']['num_threads']

    # Training
    n_steps = 128
    total_steps = n_steps * env.num_envs

    actor = ppo_module.Actor(
        ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
        ppo_module.MultivariateGaussianDiagonalCovariance(
            act_dim, env.num_envs, 1.0, NormalSampler(act_dim), cfg['seed']),
        device
    )

    critic = ppo_module.Critic(
        ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
        device
    )

    saver = ConfigurationSaver(
        log_dir=home_path + '/data/' + task_name,
        save_items=[task_path + '/cfg.yaml', task_path + '/Environment.hpp']
    )

    ppo = PPO.PPO(
        actor=actor,
        critic=critic,
        num_envs=env.num_envs,
        num_transitions_per_env=n_steps,
        num_learning_epochs=4,
        gamma=0.996,
        lam=0.95,
        num_mini_batches=4,
        device=device,
        log_dir=saver.data_dir,
        shuffle_batch=True,
    )

    if weight_path is not None:
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

    curr_rewards = np.zeros(env.num_envs)
    curr_steps = np.zeros(env.num_envs)
    episode_rewards = deque(maxlen=env.num_envs)
    episode_steps = deque(maxlen=env.num_envs)

    total_num_epochs = 5000

    for update in range(total_num_epochs):
        start = time.time()
        initial_lr = 5e-4
        lr = initial_lr - (initial_lr * (update / float(total_num_epochs)))
        for param_group in ppo.optimizer.param_groups:
            param_group['lr'] = lr

        # actual training
        for step in range(n_steps):
            action = ppo.act(obs)
            next_obs, reward, dones, info = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            obs = next_obs

            curr_rewards += reward
            curr_steps += 1
            (ids,) = np.where(dones)
            episode_rewards.extend(curr_rewards[ids])
            episode_steps.extend(curr_steps[ids])
            curr_rewards[ids] = 0
            curr_steps[ids] = 0

        # take st step to get value obs
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)

        actor.update()
        # actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

        end = time.time()

        if update % cfg['environment']['eval_every_n'] == 0:
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir + '/full_' + str(update) + '.pt')
            env.save_scaling(saver.data_dir, str(update))

        print('----------------------------------------------------')
        print(f'{update:>6}th iteration')
        print(f'average ll reward: {np.mean(episode_rewards):0.5} +- {np.std(episode_rewards):0.5f}')
        print(f'steps: {np.mean(episode_steps):0.5f} +- {np.std(episode_steps):0.5f}')
        print(f'time elapsed in this iteration: {end - start:6.4f}')
        print(f'fps: {total_steps / (end - start):6.0f}')
        print(f'real time factor: {total_steps / (end - start) * cfg["environment"]["control_dt"]:6.0f}')
        print(f'curriculum factor: {np.mean([env_info["stats"]["k_c"] for env_info in info])}')
        print('----------------------------------------------------\n')
