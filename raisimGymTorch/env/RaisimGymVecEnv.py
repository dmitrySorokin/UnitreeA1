# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os
import gym


class RaisimGymVecEnv:
    def __init__(self, impl, seed=0, update_stats=False):
        if platform.system() == 'Darwin':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.update_stats = update_stats
        self.wrapper = impl
        self.wrapper.setSeed(seed)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.wrapper.getObDim(),))
        self.action_space = gym.spaces.Box(-1, 1, (self.wrapper.getActionDim(),))

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=np.bool)
        self.wrapper.step(action, reward, done)
        return self.observe(), reward, done, self.wrapper.rewardInfo()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean = np.loadtxt(f'{dir_name}/mean{iteration}.csv', dtype=np.float32)
        var = np.loadtxt(f'{dir_name}/var{iteration}.csv', dtype=np.float32)
        self.wrapper.setObStatistics(mean, var, count)

    def save_scaling(self, dir_name, iteration):
        mean = np.zeros(self.observation_space.shape, dtype=np.float32)
        var = np.zeros(self.observation_space.shape, dtype=np.float32)
        count = 0
        self.wrapper.getObStatistics(mean, var, count)
        np.savetxt(f'{dir_name}/mean{iteration}.csv', mean)
        np.savetxt(f'{dir_name}/var{iteration}.csv', var)

    def observe(self):
        observation = np.zeros([self.num_envs, *self.observation_space.shape], dtype=np.float32)
        self.wrapper.observe(observation, self.update_stats)
        return observation

    def reset(self):
        self.wrapper.reset()
        return self.observe()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
