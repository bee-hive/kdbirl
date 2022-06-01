import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from os.path import join as pjoin
from typing import Callable, List, Tuple
import math

ASSETS_DIR = "../../gym/gym/envs/classic_control/assets"


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, m=1.0, max_steps=100, group=1, mode="train"):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = m
        self.l = 1.0
        self.viewer = None
        self.state_dim = 3
        self.unique_actions = np.arange(-self.max_torque, self.max_torque + 1e-5, 0.1)
        self.group = group
        self.step_number = 0
        self.max_steps = max_steps
        self.mode = mode

        self.theta_success_range = 12 * 2 * math.pi / 360
        self.c_trans = 0.01

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # import ipdb; ipdb.set_trace()
        u = np.clip(u, -self.max_torque, self.max_torque)
        if isinstance(u, list):
            u = u[0]
        self.last_u = u  # for rendering
        # cost = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        # if (
        #     -self.theta_success_range < th < self.theta_success_range
        # ):
        #     cost = 0
        # elif (
        #     -2 * self.theta_success_range < th < 2 * self.theta_success_range
        # ):
        #     cost = 0.1
        # else:
        #     cost = 1
        # print(angle_normalize(th))
        # cost = np.abs((th + np.pi) / (2*np.pi))
        cost = angle_normalize(th) ** 2 / (np.pi ** 2)

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self.step_number += 1

        info = {"time_limit": False}
        if self.step_number == self.max_steps:
            info["time_limit"] = True
        return self._get_obs(), cost, False, info

    def reset(self):
        # if self.mode == "train":
        high = np.array([np.pi, 1])
        # else:
        # high = np.array([0.2 * np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(
                path.dirname(__file__), pjoin(ASSETS_DIR, "clockwise.png")
            )
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def generate_tuples(self, group, n_iter=101):

        if group not in ["background", "foreground"]:
            raise Exception("group must be a string: 'background' or 'foreground'")
        # group is a string: "background" or "foreground"

        STATE_DIM = 3
        states = np.zeros((n_iter, STATE_DIM))
        s_init = self.reset()
        states[0, :] = s_init
        costs = np.ones(n_iter)
        actions = np.zeros(n_iter)

        for ii in range(1, n_iter):

            # Randomly sample an action
            a = self.action_space.sample()
            # a = np.random.choice([-2.0, -1.0, 0.0, 1.0, 2.0])
            # a = np.array([a])

            # Currently discretizes action to nearest integer
            a = np.rint(a)

            # Perform the action
            s, cost, _, _ = self.step(a)
            states[ii, :] = s
            actions[ii] = a
            costs[ii] = cost

        ## Form tuples
        tuples = []
        costs[0] = np.mean(costs)
        costs = costs - np.min(costs)
        costs = costs / np.max(costs)
        for ii in range(n_iter - 1):

            s = states[ii, :]
            a = actions[ii]
            ns = states[ii + 1, :]
            r = costs[ii]  # -costs[ii]

            # Tuples are (state, action, next state, reward, group, index)
            curr_tuple = (s, a, ns, np.asarray([r]), group, ii)
            tuples.append(curr_tuple)

        return tuples

    def generate_rollout(
        self,
        get_best_action: Callable = None,
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
            if get_best_action:
                action = get_best_action(obs)
            else:
                # action = self.action_space.sample()
                action = np.random.choice(self.unique_actions)

            next_obs, cost, done, info = self.step(action)
            rollout.append(
                (obs.squeeze(), action, cost, next_obs.squeeze(), done, group)
            )
            episode_cost += cost
            obs = next_obs
            # import ipdb; ipdb.set_trace()

            if render:
                self.render()

        return rollout, episode_cost


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":

    pend = PendulumEnv()

    STATE_DIM = 3
    n_iter = 1000
    states = np.zeros((n_iter, STATE_DIM))
    s_init = pend.reset()
    states[0, :] = s_init
    costs = np.zeros(n_iter)

    for ii in range(1, n_iter):

        # Randomly sample an action
        a = pend.action_space.sample()

        # Perform the action
        s, cost, _, _ = pend.step(a)
        states[ii, :] = s
        costs[ii] = cost
        print(cost)
        # Render the current frame
        print(cost)
        pend.render()
        # import ipdb; ipdb.set_trace()
