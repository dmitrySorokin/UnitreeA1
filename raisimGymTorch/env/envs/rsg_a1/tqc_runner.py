#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import numpy as np
import torch
import datetime
import argparse
import sys


from raisimGymTorch.algo.tqc import structures
from raisimGymTorch.algo.tqc.trainer import Trainer
from raisimGymTorch.algo.tqc.structures import Actor, Critic
from raisimGymTorch.algo.tqc.functions import eval_policy
from collections import deque
import copy


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    # task specification
    task_name = "uni_locomotion"

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
    parser.add_argument('-c', '--copy_config', help='use logged configuration in retrain mode', default=False, action='store_true')
    parser.add_argument('-i', '--iteration', help='starting iteration number', type=int, default='0')
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    args = parser.parse_args()
    mode = args.mode
    weight_path = args.weight
    copy_config = args.copy_config
    starting_iteration = args.iteration

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    log_path = task_path + "/../../../../data/" + task_name
    home_path = task_path + "/../../../../.."

    if mode == "retrain" and copy_config != False:
        weight_dir = "/".join(weight_path.split("/")[:-1])
        cfg_path = weight_dir + "/cfg.yaml"
    else:
        cfg_path = task_path + "/cfg.yaml"

    # config
    cfg = YAML().load(open(cfg_path, 'r'))

    # change k_0 value for respective starting iteration
    if starting_iteration > 0:
        k_0 = cfg['environment']['k_0']
        k_d = cfg['environment']['k_d']
        for i in range(starting_iteration):
            k_0 = k_0 ** k_d
        cfg['environment']['k_0'] = k_0

    # create environment from the configuration file
    env = VecEnv(
        rsg_a1.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper))
    )
    env.reset()

    # shortcuts
    ob_dim = env.num_obs
    act_dim = env.num_acts

    replay_buffer = structures.ReplayBuffer(ob_dim, act_dim)
    actor = Actor(ob_dim, cfg['architecture']['policy_net'], act_dim).to(device)
    actor.train()
    critic = Critic(ob_dim, cfg['architecture']['value_net'], act_dim, args.n_quantiles, args.n_nets).to(device)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets


    # Training
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * env.num_envs

    saver = ConfigurationSaver(log_dir=log_path,
                               save_items=[cfg_path, task_path + "/Environment.hpp"])
    # tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

    trainer = Trainer(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        top_quantiles_to_drop=top_quantiles_to_drop,
        discount=args.discount,
        tau=args.tau,
        target_entropy=-np.prod(act_dim).item()
    )

    for update in range(1000000):
        start = time.time()
        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.
        reward_queue = deque(maxlen=total_steps)

        if update % (cfg['environment']['eval_every_n'] * n_steps) == 0:
            print("Visualizing and evaluating the current policy")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'critic_target_state_dict': critic_target.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            env.save_scaling(saver.data_dir, str(update))

        state = env.observe(update_statistics=True)
        with torch.no_grad():
            action, _ = actor(torch.from_numpy(state).to(device))
            action = action.cpu().numpy()
        reward, dones = env.step(action)
        next_state = env.observe(update_statistics=False)
        for st, act, next_st, rew, done in zip(state, action, next_state, reward, dones):
            replay_buffer.add(st, act, next_st, rew, done)
            reward_queue.append(rew)

        # Train agent after collecting sufficient data
        if (update + 1) * env.num_envs >= args.batch_size:
            trainer.train(replay_buffer, args.batch_size)

        end = time.time()

        # update curriculum factor
        env.curriculum_callback()

        reward_ll_sum = sum(reward_queue)
        average_ll_performance = reward_ll_sum / len(reward_queue)

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(1 / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(1 / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')
