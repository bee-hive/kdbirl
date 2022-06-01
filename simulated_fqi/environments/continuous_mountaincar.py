# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
from typing import Callable, List, Tuple


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_position, goal_velocity=0, group=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.max_steps = 10
        self.goal_position = goal_position#0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.state_dim = 2
        self.unique_actions = np.arange(-1, 1 + 1e-5, 0.1)
        self.group = group
        if group == 0:
            self.power = 0.0015
        else:
            self.power = 0.007

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        position = self.state[0]
        velocity = self.state[1]
        if isinstance(action, float):
            force = min(max(action, self.min_action), self.max_action)
        else:
            force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        # Convert a possible numpy bool to a Python bool.
        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if isinstance(action, float):
            reward -= math.pow(action, 2) * 0.1
        else:
            reward -= math.pow(action[0], 2) * 0.1

        if done:
            reward = 100.0

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self):
        flip = random.randint(0, 1)
        #if flip == 0:
        self.state = np.array([self.np_random.uniform(low=-0.6, high=0.4), 0.4])
        #else:
        #    self.state = np.array([self.np_random.uniform(low=0.3, high=0.6), 0.5])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def generate_rollout(
        self,
        agent = None,
        render: bool = False,
        rollout_length: int = 50,
        group: int = 1,
    ) -> List[Tuple[np.array, int, int, np.array, bool, int]]:
        """
        Generate rollout using given action selection function.
        If a network is not given, generate random rollout instead.
        Parameters
        ----------
        get_best_action : Callable
            Greedy policy.
        render: bool
            If true, render environment.
        Returns
        -------
        rollout : List of Tuple
            Generated rollout.
        episode_cost : float
            Cumulative cost throughout the episode.
        """
        rollout = []
        episode_cost = 0
        obs = self.reset()
        info = {"time_limit": False}
        for ii in range(rollout_length):
            if agent is not None:
                action = agent.get_best_action(obs, self.unique_actions, group)
            else:
                # action = self.action_space.sample()
                action = self.action_space.sample()

            next_obs, cost, done, info = self.step(action)
            rollout.append(
                (obs.squeeze(), action, cost, next_obs.squeeze(), done, group)
            )
            episode_cost += cost
            obs = next_obs
            if done:
                break
            # import ipdb; ipdb.set_trace()

        return rollout, episode_cost
