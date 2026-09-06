import numpy as np
from collections import Counter
import random

def get_trajectories(env, Q, goal_index, goal_state, n_trajectories, n_steps,
                     source="model", P_hat=None,start_state=None):
  if start_state is None:
    start_state = env.start_state

  trajectories = []

  for _ in range(n_trajectories):
    state = start_state
    trajectory = [state]

    for _ in range(n_steps):
      if state == goal_state:
        trajectory.append(state)
        continue

      state_index = env.state_to_index[state]

      action = int(np.argmax(Q[goal_index, state_index]))

      if source == "model":
        probabilities = P_hat[state_index,action]
        next_index = random.choices(range(env.n_states), weights=probabilities, k=1)[0]
        state = env.states[next_index]

      elif source == "environment":
        state = env.move(state,action)

      else:
        raise ValueError("source must be 'model' or 'environment'")

      trajectory.append(state)

    trajectories.append(trajectory)

  return trajectories


def get_step_distributions(trajectories):
  num_steps = len(trajectories[0])
  num_trajectories = len(trajectories)
  step_distributions = []

  for step in range(num_steps):
    states_at_step = [traj[step] for traj in trajectories]
    state_counts = Counter(states_at_step)
    state_probs = {state: count / num_trajectories for state, count in state_counts.items()}

    step_distributions.append(state_probs)

  return step_distributions

def build_environment_kernel(env):
  kernel = np.zeros((env.n_states, env.n_actions, env.n_states), dtype=float)

  for state_index, state in enumerate(env.states):
    for action in range(env.n_actions):
      probabilities = env.get_transition_probabilities(state, action)
      for next_state, probability in probabilities.items():
        next_index = env.state_to_index[next_state]
        kernel[state_index, action, next_index] += probability

  return kernel


def get_exact_step_distributions(env, kernel, Q, goal_index, goal_state, n_steps, start_state=None):
    if start_state is None: 
      start_state = env.start_state
    kernel = np.asarray(kernel, dtype=float)
    policy_actions = np.argmax(Q[goal_index], axis=-1)

    policy_kernel = kernel[np.arange(env.n_states), policy_actions].copy()
    goal_state_index = env.state_to_index[goal_state]

    policy_kernel[goal_state_index] = 0.0
    policy_kernel[goal_state_index, goal_state_index] = 1.0

    distributions = np.zeros((n_steps + 1, env.n_states), dtype=float)

    start_index = env.state_to_index[start_state]
    distributions[0, start_index] = 1.0

    for step in range(n_steps):
        distributions[step + 1] = (distributions[step] @ policy_kernel)

    return distributions


def dense_distributions_to_dicts(env, distributions, tolerance=1e-15):
  return [{state: float(probability) for state, probability in zip(env.states, distribution) if probability > tolerance} for distribution in distributions]