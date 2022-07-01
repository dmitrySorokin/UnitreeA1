#!/usr/bin/env python3


import itertools
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_a1 import RaisimGymEnv
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import time
import numpy as np
import torch
import argparse
from collections import deque

from raisimGymTorch.algo.sac import SAC
from raisimGymTorch.algo.sac.replay_memory import ReplayMemory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.996, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.02, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                        help='hidden size (default: 512)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    args = parser.parse_args()

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    # directories
    # check if gpu is available
    args.cuda = True if torch.cuda.is_available() else False

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + '/../../../..'

    # config
    cfg = YAML().load(open(task_path + '/cfg.yaml', 'r'))

    saver = ConfigurationSaver(
        log_dir=home_path + '/data/anymal_locomotion',
        save_items=[task_path + '/cfg.yaml', task_path + '/Environment.hpp']
    )

    # create environment from the configuration file
    env = VecEnv(RaisimGymEnv(home_path + '/rsc', dump(cfg['environment'], Dumper=RoundTripDumper)))
    state = env.reset()
    # env = gym.make(args.env_name)
    # env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    print(agent.policy)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    # Training Loop
    updates = 0

    total_reward = deque(maxlen=env.num_envs)
    total_steps = deque(maxlen=env.num_envs)
    episode_reward = np.zeros(env.num_envs)
    episode_steps = np.zeros(env.num_envs)
    for i_step in range(5000 * 128):
        action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

        next_state, reward, done, info = env.step(action) # Step
        mask = 1.0 - done

        # update statistics
        (ids,) = np.where(done)
        episode_reward += reward
        episode_steps += 1
        total_reward.extend(episode_reward[ids])
        total_steps.extend(episode_steps[ids])
        episode_reward[ids] = 0
        episode_steps[ids] =0

        for st, act, rew, next_st, msk in zip(state, action, reward, next_state, mask):
            memory.push(st, act, rew, next_st, msk) # Append transition to memory

        state = next_state

        if i_step % 128 == 0:
            update = i_step // 128
            env.curriculum_callback()
            print(f"update: {update}, "
                  f"episode steps: {np.mean(total_steps):0.5f} +- {np.std(total_steps):0.5f}, "
                  f"reward: {np.mean(total_reward):0.5f} +- {np.std(total_reward):0.5f}"
            )

            if update % cfg['environment']['eval_every_n'] == 0:
                agent.save_checkpoint(saver.data_dir + '/full_' + str(update) + '.pt')
                env.save_scaling(saver.data_dir, str(update))

    env.close()
