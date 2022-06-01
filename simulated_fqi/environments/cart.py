"""
Modified version of classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
# flake8: noqa
import math
from typing import Callable, List, Tuple

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding


class CartEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num Observation                 Min         Max
        0   Cart Position             -4.8            4.8
        1   Cart Velocity             -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num Action
        0   Push cart to the left
        1   Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, mode="train", masscart=1.0, length=0.5, group=1, force_left=10):
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = 0.0
        self.total_mass = self.masspole + self.masscart
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.unique_actions = np.array([0, 1])
        self.group = group
        self.state_dim = 1
        self.force_left = force_left

        assert mode in ["train", "eval"]
        self.mode = mode
        self.max_steps = 300 if mode == "train" else 3000

        # Success state
        # TODO(seungjaeryanlee): Verify pole angle success state
        # NOTE(seungjaeryanlee): Relaxed definition of success state
        #                        that deviates from paper
        self.x_success_range = 1.0
        # self.theta_success_range = 12 * 2 * math.pi / 30

        # Failure state description
        # TODO(seungjaeryanlee): Verify pole angle threshold
        self.x_threshold = 2.4
        self.theta_threshold_radians = math.pi / 2

        self.c_trans = 0.01

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_next_state(self, state, action):
        x, x_dot = state
        # force = self.force_mag if action == 1 else -self.force_mag
        force = self.force_mag if action == 1 else 0
        force -= self.force_left
        # if self.group == 0:
        #     force -= 2
        # else:
        #     force -= 8
        xacc = force / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot

        return x, x_dot

    # NOTE(seungjaeryanlee): done is True only when the episode terminated due
    #                        to entering forbidden state. It is not True if it
    #                        terminated due to maximum timestep.
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        self.state = self._compute_next_state(self.state, action)
        x, _ = self.state

        self.episode_step += 1

        # Forbidden States (S-)
        # import ipdb; ipdb.set_trace()
        if (
            x < -self.x_threshold
            or x > self.x_threshold
            # or theta < -self.theta_threshold_radians
            # or theta > self.theta_threshold_radians
        ):
            done = True
            cost = 1
        # Goal States (S+)
        elif (
            -self.x_success_range
            < x
            < self.x_success_range
            # and -self.theta_success_range < theta < self.theta_success_range
        ):
            done = False
            cost = 0
            self.success_step += 1
        else:
            done = False
            cost = self.c_trans

        # Check for time limit
        info = {
            "time_limit": self.episode_step >= self.max_steps,
            "long_hold": self.success_step >= self.max_steps,
        }
        # if self.success_step > 2000:
        #     done = True

        return np.array(self.state), cost, done, info

    def reset(self):
        if self.mode == "train":
            # self.state = self.np_random.uniform(
            #     low=[-self.x_threshold, 0, 0, 0], high=[self.x_threshold, 0, 0, 0], size=(4,)
            # )
            self.state = self.np_random.uniform(
                low=[-self.x_threshold, -1], high=[self.x_threshold, 1], size=(2,)
            )
        else:
            # self.state = self.np_random.uniform(
            #     low=[-1, 0, 0, 0], high=[1, 0, 0, 0], size=(4,)
            # )
            self.state = self.np_random.uniform(
                low=[-self.x_success_range, 0],
                high=[self.x_success_range, 0],
                size=(2,),
            )
        # self.state = self.np_random.uniform(
        #         low=[-0.5, 0, 0, 0], high=[0.5, 0, 0, 0], size=(4,)
        #     )

        self.episode_step = 0
        self.success_step = 0

        return np.array(self.state)

    # def reset(self):
    #     if self.mode == "train":
    #         self.state = self.np_random.uniform(
    #             low=[-2.3, 0, -0.3, 0], high=[2.3, 0, 0.3, 0], size=(4,)
    #         )
    #     else:
    #         self.state = self.np_random.uniform(
    #             low=[-1, 0, -0.3, 0], high=[1, 0, 0.3, 0], size=(4,)
    #         )

    #     self.episode_step = 0

    #     return np.array(self.state)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            # l, r, t, b = (
            #     -polewidth / 2,
            #     polewidth / 2,
            #     polelen - polewidth / 2,
            #     -polewidth / 2,
            # )
            # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # pole.set_color(0.8, 0.6, 0.4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth / 2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(0.5, 0.5, 0.8)
            # self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            lb = screen_width // 2 - self.x_success_range * scale
            rb = screen_width // 2 + self.x_success_range * scale
            self.left_boundary = rendering.Line((lb, 0), (lb, screen_height))
            self.left_boundary.set_color(255, 0, 0)
            self.viewer.add_geom(self.left_boundary)
            self.right_boundary = rendering.Line((rb, 0), (rb, screen_height))
            self.right_boundary.set_color(255, 0, 0)
            self.viewer.add_geom(self.right_boundary)

            # self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = (
        #     -polewidth / 2,
        #     polewidth / 2,
        #     polelen - polewidth / 2,
        #     -polewidth / 2,
        # )
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def generate_rollout(
        self, get_best_action: Callable = None, render: bool = False, group: int = 1
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
        done = False
        info = {"time_limit": False, "long_hold": False}
        while not done and not info["time_limit"] and not info["long_hold"]:
            if get_best_action:
                action = get_best_action(obs)
            else:
                action = self.action_space.sample()

            next_obs, cost, done, info = self.step(action)
            a = np.zeros(2)
            a[action] = 1
            rollout.append(([obs[0]], a, cost, [next_obs[0]], done, group))
            episode_cost += cost
            obs = next_obs

            if render:
                self.render()

        return rollout, episode_cost
