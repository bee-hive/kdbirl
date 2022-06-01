#!/usr/bin/env python
import numpy as np
from pdb import set_trace


class MDP:
    def __init__(
        self, transitions, rewards, gamma, start=None, terminal=None, max_value=1000
    ):
        self.transitions = transitions
        num_actions = np.shape(transitions)[1]
        self.rewards = rewards
        self.gamma = gamma
        self.states = range(np.shape(self.transitions)[0])
        self.actions = range(np.shape(self.transitions)[1])
        self.max_value = max_value
        self.start = start
        self.terminal = len(self.states) - 1
        self.check_valid_mdp()

    def check_valid_mdp(self):
        is_valid = True
        # Need to be of the form S,A,T(S,A)
        if not (len(np.shape(self.transitions)) == 3):
            is_valid = False
        # check that state space size is same in both dims
        if not (np.shape(self.transitions)[0] == np.shape(self.transitions)[2]):
            is_valid = False
            # check that probabilities are valid
        for s in range(np.shape(self.transitions)[0]):
            for a in range(np.shape(self.transitions)[1]):
                prob_sum = 0
                for sprime in range(np.shape(self.transitions)[2]):
                    prob = self.transitions[s][a][sprime]
                    if prob < 0 or prob > 1:
                        is_valid = False
                    prob_sum += prob
                np.testing.assert_almost_equal(1.0, prob_sum)
        if self.gamma < 0 or self.gamma > 1:
            is_valid = False
        assert is_valid

    """Policy Iteration from Sutton and Barto
	   assumes discount factor of 0.99
	   Deterministic policy iteration
	"""

    def policy_iteration(self, policy=None):
        # initialization
        if policy is None:
            policy = self.get_random_policy()

        policy_stable = False
        count = 0
        while not policy_stable:
            # policy evaluation
            V = self.policy_evaluation(policy)
            count += 1
            diff_count = 0
            # policy improvement
            policy_stable = True
            for state in self.states:
                old_action = policy[state]
                # import ipdb; ipdb.set_trace()
                action_vals = np.dot(
                    self.transitions[state, :, :], self.rewards + self.gamma * V
                ).tolist()
                policy[state] = action_vals.index(max(action_vals))
                if not old_action == policy[state]:
                    diff_count += 1
                    policy_stable = False
        return (policy, V)

    def get_random_policy(self):
        policy = np.zeros(len(self.states), dtype=np.uint32)
        for state in self.states:
            policy[state] = np.random.randint(0, len(self.actions))
        return policy

    """
	policy - deterministic policy, maps state to action
	-Deterministic policy evaluation
	"""

    def policy_evaluation(self, policy, theta=0.0001):
        V = np.zeros(len(self.states))
        count = 0
        while True:
            delta = 0
            for state in self.states:
                value = V[state]
                V[state] = np.dot(
                    self.transitions[state, policy[state], :],
                    self.rewards + self.gamma * V,
                )
                delta = max(delta, np.abs(value - V[state]))
                # If divergence and policy has value -inf, return value function early
                if V[state] > self.max_value:
                    return V
                if V[state] < -self.max_value:
                    return V
            if delta < theta:
                break
        return V

    def policy_q_evaluation(self, policy):
        V = self.policy_evaluation(policy)
        Q = np.zeros(np.shape(self.transitions)[0:2])
        for state in self.states:
            for action in self.actions:
                Q[state, action] = np.dot(
                    self.transitions[state, action, :], self.rewards + self.gamma * V
                )
        return Q

    def value_iteration(self, theta=0.0001):
        V = np.zeros(len(self.states))
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                V[state] = np.amax(
                    np.dot(self.transitions[state, :, :], self.rewards + self.gamma * V)
                )
                delta = max(delta, np.abs(v - V[state]))
            if delta < theta:
                break
        return V

    def q_value_iteration(self, theta=0.0001):
        Q = np.zeros((len(self.states), len(self.actions)))
        while True:
            delta = 0
            for state in self.states:
                for action in self.actions:
                    q = Q[state, action]
                    Q[state, action] = np.dot(
                        self.transitions[state, action, :],
                        self.rewards + self.gamma * np.amax(Q, axis=1),
                    )
                    delta = max(delta, np.abs(q - Q[state, action]))
            if delta < theta:
                break
        return Q

    """
	Takes a state and action, returns next state
	Chooses a next state according to transition 
	probabilities
	"""

    def act(self, state, action):
        next_state = np.random.choice(self.states, p=self.transitions[state, action, :])
        return self.rewards[next_state], next_state
