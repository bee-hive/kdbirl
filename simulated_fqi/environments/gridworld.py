import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
import random


class Gridworld(gym.Env):
    def __init__(self, group, shared_weights=None, fg_weights=None, target_state=None, gridsize=4, nonlinear=False, parametric=False,
                 reward_params=None):
        self.unique_actions = [1, 2, 3, 4]
        self.action_space = spaces.Discrete(4)
        self.shared_weights = shared_weights
        self.fg_weights = fg_weights
        # Removed not moving, makes it too slow and confusing
        # 1 (move left), 2 (move right), 3 (move down), 4 (move up)
        self.action_pos_dict = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
        self.group = group

        """ set observation space """
        self.n = gridsize

        """ agent state: start, target, current state """
        self.agent_start_state = [0, 0]
        
        self.agent_target_state = target_state
        self.agent_state = copy.deepcopy(self.agent_start_state)

        """ weights """
        self.shared_weights = shared_weights
        self.fg_weights = fg_weights
        self.nonlinear = nonlinear
        self.parametric= parametric
        self.reward_params=reward_params

    def step(self, action):
        """return next observation, reward, finished, success"""
        action = int(action)
        info = {}
        nxt_agent_state = (
            self.agent_state[0] + self.action_pos_dict[action][0],
            self.agent_state[1] + self.action_pos_dict[action][1],
        )
        self.agent_state = nxt_agent_state
        # If out of bounds
        if (
            nxt_agent_state[0] < 0
            or nxt_agent_state[0] >= self.n
            or nxt_agent_state[1] < 0
            or nxt_agent_state[1] >= self.n
        ):
            return (nxt_agent_state, -10, False, info)

        # nxt_agent_state is a valid state
        if self.nonlinear:
            if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]:
                reward = 1.0
            else:
                reward = 0.0
        elif self.parametric:
            reward = np.power(nxt_agent_state[0], 2)*self.reward_params[0] + nxt_agent_state[0]*nxt_agent_state[1]*self.reward_params[1] + self.reward_params[2]
        else:
            if self.group == 0:
                reward = np.dot(nxt_agent_state, self.shared_weights)
            elif self.group == 1:
                reward = np.dot(nxt_agent_state, np.add(self.shared_weights, self.fg_weights))
        if (nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]):
            info["success"] = True
            return (nxt_agent_state, reward, True, info)
        return (nxt_agent_state, reward, False, info)

    def reset(self):
        self.agent_state = copy.deepcopy(self.agent_start_state)
        return self.agent_state         

    def reset_random(self):
        coords = np.arange(0, self.n-1)
        self.agent_state = [random.choice(coords), random.choice(coords)]
        return self.agent_state

    def get_agent_state(self):
        """get current agent state"""
        return self.agent_state

    def get_start_state(self):
        """get current start state"""
        return self.agent_start_state

    def get_target_state(self):
        """get current target state"""
        return self.agent_target_state

    def _close_env(self):
        plt.close(1)
        return

    def find_valid_actions(self, pos):
        valid_actions = []
        for k in self.action_pos_dict:
            nxt_agent_state = (
                pos[0] + self.action_pos_dict[k][0],
                pos[1] + self.action_pos_dict[k][1],
            )
            if (
                nxt_agent_state[0] >= 0
                and nxt_agent_state[0] < self.n
                and nxt_agent_state[1] >= 0
                and nxt_agent_state[1] < self.n
            ):
                valid_actions.append(k)
        return valid_actions

    def generate_rollout(
        self,
        agent=None,
        render: bool = False,
        group: int = 1,
        random_start: bool = True,
    ) -> List[Tuple[np.array, int, int, np.array, bool, int]]:
        """
        Generate rollout using given action selection function.
        If a network is not given, generate random rollout instead.
        Parameters
        ----------
        agent : NFQAgent
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
        if random_start:
            if random.choice([0, 1]) == 0:
                obs = self.reset()
            else:
                obs = self.reset_random()
        else:
            obs = self.reset()
        time_limit = 50
        it = 0
        while it < time_limit:
            if agent:
                valid_actions = self.find_valid_actions(obs)
                action = agent.get_best_action(obs, valid_actions, group=group)
            else:
                valid_actions = self.find_valid_actions(obs)
                action = random.choice(valid_actions)
            next_obs, cost, done, info = self.step(action)
            rollout.append((obs, action, cost, next_obs, done, group))
            episode_cost += cost
            obs = next_obs
            it += 1
            if done:
                rollout.append((self.agent_target_state, [0, 0], cost, self.agent_target_state, True, group))
                break

        return rollout, episode_cost
