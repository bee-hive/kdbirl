import seaborn as sns
from simulated_fqi.environments.gridworld import Gridworld
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import tqdm
import scipy
import pickle
import math
from irl.original_birl.birl.main import convert_state_coord_to_state_idx, convert_state_idx_to_state_coord



def generate_policy_rollout_nonlinear(group=0, target_state=None, gridsize=10):
    rollout = []
    # obs = [np.random.choice(gridsize), np.random.choice(gridsize)]
    obs = [0, 0]
    time_limit = 100
    it = 0
    done = False
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)

        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            if nxt_agent_state[0] == target_state[0] and nxt_agent_state[1] == target_state[1]:
                reward = 1.0
            else:
                reward = 0.0
            curr_rewards[ii] = reward + np.random.normal(scale=0.01)

        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = (obs[0] + action[0], obs[1] + action[1])
        cost = max_reward
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, group))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            break
    return rollout, done

def convert_state_to_state_idx(state_coord, grid_dimension):
	return grid_dimension * state_coord[0] + state_coord[1]

def generate_policy_rollout_nonparametric(reward_vec, target_state, gridsize=10, time_limit=25):
    rollout = []
    obs = [0, 0]
    it = 0
    done = False
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)
        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            nxt_state_idx = convert_state_to_state_idx(nxt_agent_state, gridsize)
            reward = reward_vec[nxt_state_idx]
            curr_rewards[ii] = reward + np.random.normal(scale=0.01)

        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = (obs[0] + action[0], obs[1] + action[1])
        cost = max_reward
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, 0))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, [0, 0], 1.0, target_state, True, 0))
            break
    return rollout, done


def runNonlinearFQI(behavior=True, group=0, target_state=None, init_experience=100, n=4, num_rollouts=100):
    train_env = Gridworld(group=group, target_state=target_state,
                          gridsize=n, nonlinear=True)

    rollouts = []
    if init_experience > 0:
        for _ in range(init_experience):
            rollout, episode_cost = train_env.generate_rollout(None, render=False, group=group, random_start=True)
            rollouts.extend(rollout)
    all_rollouts = rollouts.copy()

    if not behavior:
        state_b, action_b, cost_b, next_state_b, done_b, group_b = zip(*all_rollouts)
        cost_b = torch.FloatTensor(cost_b)
        reg = LinearRegression().fit(next_state_b, cost_b)
        rollouts = []
        if init_experience > 0:
            for _ in range(int(init_experience / 2)):
                rollout = generate_rollout(reg, group=group, target_state=target_state)
                rollouts.extend(rollout)
        policy_rollouts = rollouts
    else:
        rollouts = []
        if init_experience > 0:
            for _ in range(num_rollouts):
                rollout, done = generate_policy_rollout_nonlinear(group=group, target_state=target_state, gridsize=n)
                if done:
                    rollouts.extend(rollout)
            policy_rollouts = rollouts
    return policy_rollouts

def runNonparametricFQI(reward_vec, gridsize=2, num_rollouts=100):
    # Find the target state
    max_reward_idx = np.argmax(reward_vec)
    target_state = convert_state_idx_to_state_coord(max_reward_idx, gridsize)

    rollouts = []
    for _ in range(num_rollouts):
        rollout, done = generate_policy_rollout_nonparametric(reward_vec, target_state=target_state, gridsize=gridsize)
        if done:
            rollouts.extend(rollout)
    policy_rollouts = rollouts
    return policy_rollouts


def plot_reward(n=4, title='Reward = x + y', reward_params=None):
    reward_matrix = np.zeros((n, n))
    positions = [i for i in range(n)]
    for i, x in enumerate(range(n)):
        for j, y in enumerate(range(n)):
            reward = x * reward_params[0] + y * reward_params[1]
            reward_matrix[j, i] = reward
    plt.figure(figsize=(5, 5))
    reward_matrix /= np.max(np.abs(reward_matrix))
    ax = sns.heatmap(reward_matrix, xticklabels=positions, yticklabels=positions)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.invert_yaxis()


def estimate_expert_posterior(r_k, behavior_opt, observations, metric_r='cosine'):
    post = 0
    dist_rewards = distance_rewards(r_k, observations, metric_r=metric_r)
    for s_i in behavior_opt:
        # Sum
        sum_si = 0
        for s_j in observations:
            if metric_r == "cosine":
                weight = distance_r(r_k, s_j[2]) / dist_rewards
            elif metric_r == "euclidean":
                weight = distance_r_euclidean(r_k, s_j[2]) / dist_rewards
            likelihood = distance_points(s_i, s_j) * weight
            sum_si += likelihood
        if sum_si == 0:
            post += np.log(0.000000000001)
        else:
            post += np.log(sum_si)
    return post

def estimate_expert_posterior_nonparametric(reward_vector, behavior_opt, observations):
    post = 0
    dist_rewards = distance_rewards_nonparametric(reward_vector, observations, math.sqrt(len(reward_vector)), linear=True)
    for s_i in behavior_opt:
        # Sum
        sum_si = 0
        for s_j in observations:
            s_j_r = s_j[2] #targetstate_to_rewardvec(s_j[2], math.sqrt(len(reward_vector)))
            weight = distance_r_nonparametric(reward_vector, s_j_r) / dist_rewards
            likelihood = distance_points(s_i, s_j) * weight
            sum_si += likelihood
        if sum_si == 0:
            post += np.log(0.000000000001)
        else:
            post += np.log(sum_si)
    return post


def find_valid_actions(pos, gridsize=4):
    valid_actions = []
    action_pos_dict = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    for k in action_pos_dict:
        nxt_agent_state = (
            pos[0] + action_pos_dict[k][0],
            pos[1] + action_pos_dict[k][1],
        )
        if (
            nxt_agent_state[0] >= 0
            and nxt_agent_state[0] < gridsize
            and 0 <= nxt_agent_state[1] < gridsize
        ):
            valid_actions.append(k)
    return valid_actions, action_pos_dict


def generate_rollout(reg, group=0, target_state=None, gridsize=4):
    rollout = []
    obs = [np.random.choice(4), np.random.choice(4)]
    time_limit = 15
    it = 0
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)

        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            reward = reg.predict([[nxt_agent_state[0], nxt_agent_state[1]]])
            curr_rewards[ii] = reward + np.random.normal(scale=0.05)

        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = (obs[0] + action[0], obs[1] + action[1])
        cost = max_reward
        done = False
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, group))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, 0, max_reward, target_state, done, group))
            break
    return rollout


def generate_policy_rollout(group=0, target_state=None, true_weights_shared=None, true_weights_fg=None, gridsize=4, random_start=False):
    rollout = []
    if random_start:
        obs = [np.random.choice(gridsize), np.random.choice(gridsize)]
    else:
        obs = [0, 0]
    time_limit = 5
    it = 0
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)
        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            if group == 0:
                reward = np.dot(nxt_agent_state, true_weights_shared)
            elif group == 1:
                reward = np.dot(nxt_agent_state, np.add(true_weights_shared, true_weights_fg))
            curr_rewards[ii] = reward + np.random.normal(scale=0.01)

        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = (obs[0] + action[0], obs[1] + action[1])
        cost = max_reward
        done = False
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, group))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, [0, 0], np.dot(target_state, true_weights_shared), target_state, True, group))
            rollout.append((target_state, [0, 0], np.dot(target_state, true_weights_shared), target_state, True, group))
            break
    return rollout


def find_target_state(reward_weights, gridsize=4):
    target_state = None
    best_reward = None
    for x in range(0, gridsize):
        for y in range(0, gridsize):
            state = [x, y]
            r = np.dot(state, reward_weights)
            if best_reward is None or r > best_reward:
                target_state = state
                best_reward = r
    # if target_state[0] == 0:
    #     target_state[0] = target_state[1]
    # if target_state[1] == 0:
    #     target_state[1] = target_state[0]
    return target_state


def runLinearFQI(init_experience=50, num_rollouts=10, reward_weights_shared=np.asarray([3, 3]),
                 reward_weights_fg=np.asarray([-6, 0]), dataset='bg', q_val=False, behavior=False, gridsize=4):
    if dataset == 'bg':
        group = 0
    else:
        group = 1

    target_state = find_target_state(reward_weights_shared, gridsize=gridsize)
    train_env = Gridworld(group=group, shared_weights=reward_weights_shared, fg_weights=reward_weights_fg, target_state=target_state,
                          gridsize=gridsize)

    rollouts = []
    if init_experience > 0:
        for _ in range(init_experience):
            rollout, episode_cost = train_env.generate_rollout(None, render=False, group=group, random_start=True)
            rollouts.extend(rollout)
    all_rollouts = rollouts.copy()

    if not behavior:
        gamma = 0.95
        state_b, action_b, cost_b, next_state_b, done_b, group_b = zip(*all_rollouts)
        done_b = torch.FloatTensor(done_b)
        cost_b = torch.FloatTensor(cost_b)
        if q_val:
            target_q_values = cost_b.squeeze() + gamma * (1 - done_b)
            reg = LinearRegression().fit(next_state_b, target_q_values)
        else:
            reg = LinearRegression().fit(next_state_b, cost_b)
        rollouts = []
        if init_experience > 0:
            for _ in range(int(init_experience / 2)):
                rollout = generate_rollout(reg, group=group, target_state=target_state, gridsize=gridsize)
                rollouts.extend(rollout)
        policy_rollouts = rollouts
        return policy_rollouts, reg
    else:
        rollouts = []
        for _ in range(int(num_rollouts)):
            rollout = generate_policy_rollout(group=group, target_state=target_state, true_weights_shared=reward_weights_shared, gridsize=gridsize,
                                              random_start=True)
            rollouts.extend(rollout)
        policy_rollouts = rollouts
        return policy_rollouts, None


def conditional_dist(s_i, dataset, reward, metric_r='cosine'):
    sum = 0
    dist_rewards = distance_rewards(reward, dataset, metric_r=metric_r)
    for s_j in dataset:
        if metric_r == 'cosine':
            weight = distance_r(reward, s_j[2]) / dist_rewards
        elif metric_r == 'euclidean':
            weight = distance_r_euclidean(reward, s_j[2]) / dist_rewards
        dist = distance_points(s_i, s_j)
        est = dist * weight
        sum += est
    return sum

def targetstate_to_rewardvec(target_state, gridsize):
    reward_vec = np.zeros(int(gridsize*gridsize))
    idx = target_state[0]*gridsize + target_state[1]
    reward_vec[int(idx)] = 1
    return reward_vec

def linearreward_to_rewardvec(linearreward, gridsize):
    gridsize = int(gridsize)
    reward_vec = np.zeros((gridsize, gridsize))
    for x in range(gridsize):
        for y in range(gridsize):
            reward_vec[x, y] = np.dot(linearreward, [x, y])
    return reward_vec.flatten()

def distance_rewards_nonparametric(reward_vector, observations, gridsize, linear):
    sum_diff = 0
    for sample in observations:
        if linear:
            if len(reward_vector) == 2:
                reward_vec = linearreward_to_rewardvec(sample[2], gridsize)
            else:
                reward_vec = reward_vector
        else:
            ts = sample[2]
            reward_vec = targetstate_to_rewardvec(ts, gridsize)
        sum_diff += np.linalg.norm(reward_vector-reward_vec)
    return sum_diff

def distance_r_nonparametric(r, r_prime):
    h_prime = 0.5
    dist = np.linalg.norm(r-r_prime)
    return np.exp(-(np.power(dist, 2) / (2 * h_prime)))

def conditional_dist_nonparametric(state, dataset, correct_reward, linear):
    sum = 0
    dist_rewards = distance_rewards_nonparametric(correct_reward, dataset, int(math.sqrt(len(correct_reward))), linear=linear)
    import ipdb; ipdb.set_trace()
    for s_j in dataset:
        if linear:
            s_j_r = linearreward_to_rewardvec(s_j[2], int(math.sqrt(len(correct_reward))))
        else:
            s_j_r = targetstate_to_rewardvec(s_j[2], math.sqrt(len(correct_reward)))
        weight = distance_r_nonparametric(correct_reward, s_j_r) / dist_rewards
        dist = distance_points(state, s_j)
        est = dist * weight
        sum += est
    return sum


# Used for linear reward function with reward parameterized as vector of length 2
def heatmap_dataset_conditional_density_posterior(reward, observations, gridsize, plot_loc=".", m=300, n=50):
    # m expert observations
    behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=10, behavior=True, reward_weights_shared=reward, gridsize=gridsize)

    idx = [i for i in range(len(observations))]
    size_obs = min(m, len(observations))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    observations = np.asarray(observations)[new_idx]

    idx = [i for i in range(len(behavior_opt))]
    size_behav = min(n, len(behavior_opt))
    new_idx = np.random.choice(idx, size=size_behav, replace=False)
    behavior_opt = np.asarray(behavior_opt)[new_idx]

    heatmap_dataset = generate_dataset_plot(behavior_opt, gridsize=gridsize)
    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), constrained_layout=True)
    sns.heatmap(heatmap_dataset, xticklabels=x, yticklabels=y, ax=axs[0])
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Demonstration density for reward: " + str(reward))
    axs[0].invert_yaxis()

    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_conditional = np.zeros((gridsize, gridsize))
    action_pos_dict = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    conditional_est = []
    for ll, a in enumerate(tqdm.tqdm([1, 2, 3, 4])):
        action = action_pos_dict[a]
        for kk, i in enumerate(range(xx.shape[1])):
            for j in range(yy.shape[0]):
                # Evaluate the posterior at this reward parameter
                state = ([xx[0][i], yy[j][0]], action)
                c_est = conditional_dist(state, observations, reward)
                conditional_est.append(c_est)
                heatmap_conditional[j, i] += c_est

    sns.heatmap(heatmap_conditional, xticklabels=x, yticklabels=y, ax=axs[1])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Grid X")
    axs[1].set_ylabel("Grid Y")
    axs[1].set_title("Conditional density wrt reward: " + str(reward))

    x = np.arange(-1, 1.1, 0.2)
    x = np.around(x, decimals=1)
    y = np.arange(-1, 1.1, 0.2)
    y = np.around(y, decimals=1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_posterior = np.zeros((11, 11))
    for kk, i in enumerate(tqdm.tqdm(range(xx.shape[1]))):
        for j in range(yy.shape[0]):
            # Evaluate the posterior at this reward parameter
            r_k = [xx[0][i], yy[j][0]]
            post = estimate_expert_posterior(r_k, behavior_opt, observations)
            heatmap_posterior[j, i] = post
    sns.heatmap(heatmap_posterior, xticklabels=x, yticklabels=y, ax=axs[2])
    axs[2].invert_yaxis()
    axs[2].set_xlabel("Reward parameter 1")
    axs[2].set_ylabel("Reward parameter 2")
    axs[2].set_title("Expert posterior, true reward=" + str(reward))

    file = plot_loc + "r=" + str(reward) + "_m=" + str(m) + "_n=" + str(n)
    plt.savefig(file + ".png")
    plt.close()

    np.save(file + ".npy", heatmap_posterior)
    return heatmap_posterior

# Used for linear reward function with reward parameterized as a vector of length gridsize*gridsize
def heatmap_dataset_conditional_density_posterior_nonparametric(reward, observations, gridsize, plot_loc=".", m=300, n=50):
    # m expert observations
    behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=10, behavior=True, reward_weights_shared=reward, gridsize=gridsize)

    idx = [i for i in range(len(observations))]
    size_obs = min(m, len(observations))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    observations = np.asarray(observations)[new_idx]

    idx = [i for i in range(len(behavior_opt))]
    size_behav = min(n, len(behavior_opt))
    new_idx = np.random.choice(idx, size=size_behav, replace=False)
    behavior_opt = np.asarray(behavior_opt)[new_idx]

    #print(str(find_optimal_bandwidth(observations, gridsize, "nonparametric")))
    heatmap_dataset = generate_dataset_plot(behavior_opt, gridsize=gridsize)
    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), constrained_layout=True)
    sns.heatmap(heatmap_dataset, xticklabels=x, yticklabels=y, ax=axs[0])
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Demonstration density for reward: " + str(reward))
    axs[0].invert_yaxis()

    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_conditional = np.zeros((gridsize, gridsize))
    conditional_est = []
    reward_vec = linearreward_to_rewardvec(reward, gridsize)
    print("Reward vec: ", reward_vec)
    import ipdb; ipdb.set_trace()
    for kk, i in enumerate(range(xx.shape[1])):
        for j in range(yy.shape[0]):
            # Evaluate the posterior at this reward parameter
            state = ([xx[0][i], yy[j][0]])
            c_est = conditional_dist_nonparametric(state, observations, reward_vec, linear=True)
            conditional_est.append(c_est)
            heatmap_conditional[j, i] += c_est

    sns.heatmap(heatmap_conditional, xticklabels=x, yticklabels=y, ax=axs[1])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Grid X")
    axs[1].set_ylabel("Grid Y")
    axs[1].set_title("Conditional density wrt reward: " + str(reward))

    # x = np.arange(-1, 1.1, 0.2)
    # x = np.around(x, decimals=1)
    # y = np.arange(-1, 1.1, 0.2)
    # y = np.around(y, decimals=1)
    # xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_posterior = np.zeros((11, 11))
    # for kk, i in enumerate(tqdm.tqdm(range(xx.shape[1]))):
    #     for j in range(yy.shape[0]):
    #         # Evaluate the posterior at this reward parameter
    #         r_k = [xx[0][i], yy[j][0]]
    #         reward_vec = linearreward_to_rewardvec(r_k, gridsize)
    #         post = estimate_expert_posterior_nonparametric(reward_vec, behavior_opt, observations)
    #         heatmap_posterior[j, i] = post
    # # TODO: can't visualize these rewards like this anymore.
    # sns.heatmap(heatmap_posterior, xticklabels=x, yticklabels=y, ax=axs[2])
    # axs[2].invert_yaxis()
    # axs[2].set_xlabel("Reward parameter 1")
    # axs[2].set_ylabel("Reward parameter 2")
    # axs[2].set_title("Expert posterior, true reward=" + str(reward))

    file = plot_loc + "r=" + str(reward) + "_m=" + str(m) + "_n=" + str(n) + "_nonparametric"
    plt.savefig(file + ".png")
    plt.close()

    np.save(file + ".npy", heatmap_posterior)
    return heatmap_posterior

# Make the training reward functions evenly distributed
def generate_observations(init_experience=5, gridsize=4):
    rewards = []
    for i in range(-10, 11, 3):
        for j in range(-10, 11, 3):
            rewards.append([i / 10, j / 10])

    observations = []
    for r in rewards:
        behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=init_experience, behavior=True, reward_weights_shared=r,
                                               gridsize=gridsize)
        for sample in behavior_opt:
            s = (sample[0], sample[1], r)
            observations.append(s)
    return observations


def distance_r(r, r_prime):
    h_prime = 0.0012  # Proportional to standard deviation of all reward function distances (or variance)
    #     h_prime = 0.001 # Proportional to mean
    h_prime = 0.05  # 149
    #     h=10e-3
    dist = scipy.spatial.distance.cosine(r, r_prime)
    return np.exp(-(np.power(dist, 2) / (2 * h_prime)))


def distance_r_euclidean(r, r_prime):
    h_prime = 0.2
    h_prime = 0.16141846455350892
    dist = scipy.spatial.distance.euclidean(r, r_prime)
    # Euclidean distance
    return np.exp(-(np.power(dist, 2) / (2 * h_prime)))


def distance_points(p1, p2):
    h = 0.19
    # Removing the action part
    dist = scipy.spatial.distance.euclidean(p1[0], p2[0])  # + scipy.spatial.distance.euclidean(p1[1], p2[1])
    return np.exp(-np.power(dist, 2) / (2 * h))


def distance_rewards(r_k, observations, metric_r='cosine'):
    sum_diff = 0
    for sample in observations:
        r = sample[2]
        if metric_r == 'cosine':
            sum_diff += distance_r(r_k, r)
        elif metric_r == 'euclidean':
            sum_diff += distance_r_euclidean(r_k, r)
    return sum_diff


def find_optimal_bandwidth(observations, gridsize, metric_r):
    # Distance between rewards, h_prime
    all_distances_r = []
    all_distances_p = []
    for ii, o in enumerate(tqdm.tqdm(observations)):
        for o_prime in observations:
            if metric_r == 'cosine':
                all_distances_r.append(distance_r(o[1], o_prime[1]))
            elif metric_r == 'euclidean':
                all_distances_r.append(distance_r_euclidean(o[1], o_prime[1]))
            elif metric_r == 'nonparametric':
                r = linearreward_to_rewardvec(o[1], gridsize)
                r_prime = linearreward_to_rewardvec(o_prime[1], gridsize)
                all_distances_r.append(distance_r_nonparametric(r, r_prime))
            all_distances_p.append(distance_points(o[0], o_prime[0]))
    h_prime = np.std(all_distances_r) * np.std(all_distances_r)
    h = np.std(all_distances_p) * np.std(all_distances_p)
    return h, h_prime


def generate_dataset_plot(dataset, gridsize=4):
    xs = [b[0][0] for b in dataset]
    ys = [b[0][1] for b in dataset]

    heatmap_dataset = np.zeros((gridsize, gridsize))
    for i, x_i in enumerate(xs):
        y_i = ys[i]
        heatmap_dataset[y_i, x_i] += 1

    heatmap_dataset /= np.max(np.abs(heatmap_dataset))
    return heatmap_dataset


def most_common_reward(hmap_posterior, top=True, pct=0.1, n=11):
    hmap_posterior[5][5] = np.nanmin(hmap_posterior)
    post = hmap_posterior / np.nanmax(np.abs(hmap_posterior))
    post = np.exp(post)
    post /= np.sum(post)

    rewards = np.random.multinomial(1000000, post.flatten())
    rewards = rewards.reshape((n, n))

    evaluations = rewards // 100

    x = np.arange(-1, 1.1, 0.2)
    x = np.around(x, decimals=1)
    y = np.arange(-1, 1.1, 0.2)
    y = np.around(y, decimals=1)
    xx, yy = np.meshgrid(x, y, sparse=True)

    reward_fns = []
    for kk, i in enumerate(range(n)):
        for ll, j in enumerate(range(n)):
            r_k = [xx[0][i], yy[j][0]]
            if r_k == [0, 0]:
                continue
            count = evaluations[ll, kk]
            for aa in range(count):
                reward_fns.append(r_k)
    fn_to_count = {}
    for fn in reward_fns:
        if tuple(fn) not in fn_to_count:
            fn_to_count[tuple(fn)] = 0
        fn_to_count[tuple(fn)] += 1

    fns = sorted(fn_to_count.items(), key=lambda item: item[1], reverse=True)
    if top:
        most_common_fns = fns[:int(len(fns) * pct)]
    else:
        most_common_fns = fns[int(len(fns) * pct):]
    return most_common_fns

# Used for step reward functions with reward as a target state
def heatmap_dataset_conditional_density_posterior_nonlinear(target_state, observations, gridsize=10, plot_loc=".", m=300, n=50):
    # datadir = "/tigress/BEE/bayesian_irl/datasets/grid_datasets/"
    # behavior_opt = pickle.load(open(datadir + "rollouts_target=" + str(target_state) + ".pkl", "rb"))
    behavior_opt = runNonlinearFQI(init_experience=200, behavior=True, target_state=target_state, n=2, num_rollouts=100)

    # Filter the number of samples
    idx = [i for i in range(len(observations))]
    size_obs = min(m, len(observations))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    observations = np.asarray(observations)[new_idx]

    print(str(find_optimal_bandwidth(observations, metric_r='euclidean')))

    idx = [i for i in range(len(behavior_opt))]
    size_behav = min(n, len(behavior_opt))
    new_idx = np.random.choice(idx, size=size_behav, replace=False)
    behavior_opt = np.asarray(behavior_opt)[new_idx]

    heatmap_dataset = generate_dataset_plot(behavior_opt, gridsize=gridsize)

    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), constrained_layout=True)
    sns.heatmap(heatmap_dataset, xticklabels=x, yticklabels=y, ax=axs[0])
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Demonstration density for target: " + str(target_state))
    axs[0].invert_yaxis()

    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_conditional = np.zeros((gridsize, gridsize))
    action_pos_dict = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    for ll, a in enumerate(tqdm.tqdm([1, 2, 3, 4])):
        action = action_pos_dict[a]
        for kk, i in enumerate(range(gridsize)):
            for j in range(gridsize):
                # Evaluate the posterior at this reward parameter
                state = ([xx[0][i], yy[j][0]], action)
                c_est = conditional_dist(state, observations, target_state, metric_r='euclidean')
                heatmap_conditional[j, i] += c_est

    sns.heatmap(heatmap_conditional, xticklabels=x, yticklabels=y, ax=axs[1], vmax=0.4)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Grid X")
    axs[1].set_ylabel("Grid Y")
    axs[1].set_title("Conditional density wrt target: " + str(target_state))

    x = np.arange(0, gridsize, 1)
    x = np.around(x, decimals=1)
    y = np.arange(0, gridsize, 1)
    y = np.around(y, decimals=1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_posterior = np.zeros((gridsize, gridsize))
    for kk, i in enumerate(tqdm.tqdm(range(gridsize))):
        for j in range(gridsize):
            # Evaluate the posterior at this reward parameter
            ts = [xx[0][i], yy[j][0]]
            post = estimate_expert_posterior(ts, behavior_opt, observations, metric_r='euclidean')
            heatmap_posterior[j, i] = post

    sns.heatmap(heatmap_posterior, xticklabels=x, yticklabels=y, ax=axs[2])
    axs[2].invert_yaxis()
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    axs[2].set_title("Expert posterior, true target=" + str(target_state))

    file = plot_loc + "target=" + str(target_state) + "_m=" + str(m) + "_n=" + str(n)
    plt.savefig(file + ".png")
    np.save(file + ".npy", heatmap_posterior)
    plt.close()

# Used for step reward functions with reward as a vector, not a target state
def heatmap_dataset_conditional_density_posterior_nonlinear_nonparametric(target_state, observations, gridsize=10, plot_loc=".", m=300, n=50):
    # datadir = "/tigress/BEE/bayesian_irl/datasets/grid_datasets/"
    # behavior_opt = pickle.load(open(datadir + "rollouts_target=" + str(target_state) + ".pkl", "rb"))
    behavior_opt = runNonlinearFQI(init_experience=200, behavior=True, target_state=target_state, n=2, num_rollouts=100)

    # Filter the number of samples
    idx = [i for i in range(len(observations))]
    size_obs = min(m, len(observations))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    observations = np.asarray(observations)[new_idx]

    idx = [i for i in range(len(behavior_opt))]
    size_behav = min(n, len(behavior_opt))
    new_idx = np.random.choice(idx, size=size_behav, replace=False)
    behavior_opt = np.asarray(behavior_opt)[new_idx]

    heatmap_dataset = generate_dataset_plot(behavior_opt, gridsize=gridsize)

    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), constrained_layout=True)
    sns.heatmap(heatmap_dataset, xticklabels=x, yticklabels=y, ax=axs[0])
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Demonstration density for target: " + str(target_state))
    axs[0].invert_yaxis()

    # R = the correct reward function, pass this in instead of the target_state
    x = np.arange(0, gridsize, 1)
    y = np.arange(0, gridsize, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    correct_reward = targetstate_to_rewardvec(target_state, gridsize)
    heatmap_conditional = np.zeros((gridsize, gridsize))
    action_pos_dict = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    for ll, a in enumerate(tqdm.tqdm([1, 2, 3, 4])):
        action = action_pos_dict[a]
        for kk, i in enumerate(range(gridsize)):
            for j in range(gridsize):
                # Evaluate the posterior at this reward parameter
                state = ([xx[0][i], yy[j][0]], action)
                c_est = conditional_dist_nonparametric(state, observations, correct_reward)
                heatmap_conditional[j, i] += c_est

    sns.heatmap(heatmap_conditional, xticklabels=x, yticklabels=y, ax=axs[1], vmax=0.4)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Grid X")
    axs[1].set_ylabel("Grid Y")
    axs[1].set_title("Conditional density wrt target: " + str(target_state))

    # TODO: not correct anymore, the visualization should be over the reward functions, not the states.
    x = np.arange(0, gridsize, 1)
    x = np.around(x, decimals=1)
    y = np.arange(0, gridsize, 1)
    y = np.around(y, decimals=1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    heatmap_posterior = np.zeros((gridsize, gridsize))
    for kk, i in enumerate(tqdm.tqdm(range(gridsize))):
        for j in range(gridsize):
            # Evaluate the posterior at this reward parameter
            ts = [xx[0][i], yy[j][0]]
            reward_vector = targetstate_to_rewardvec(ts, gridsize)
            post = estimate_expert_posterior_nonparametric(reward_vector, behavior_opt, observations)
            heatmap_posterior[j, i] = post

    sns.heatmap(heatmap_posterior, xticklabels=x, yticklabels=y, ax=axs[2])
    axs[2].invert_yaxis()
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    axs[2].set_title("Expert posterior, true target=" + str(target_state))

    file = plot_loc + "target=" + str(target_state) + "_m=" + str(m) + "_n=" + str(n)
    plt.savefig(file + ".png")
    np.save(file + ".npy", heatmap_posterior)
    plt.close()

def sample_from_posterior():
    pass
