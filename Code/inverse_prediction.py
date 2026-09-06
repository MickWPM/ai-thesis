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


# def get_exact_step_distributions(env, kernel, Q, goal_index, goal_state, n_steps, start_state=None):
#     if start_state is None: 
#       start_state = env.start_state
#     kernel = np.asarray(kernel, dtype=float)
#     policy_actions = np.argmax(Q[goal_index], axis=-1)

#     policy_kernel = kernel[np.arange(env.n_states), policy_actions].copy()
#     goal_state_index = env.state_to_index[goal_state]

#     policy_kernel[goal_state_index] = 0.0
#     policy_kernel[goal_state_index, goal_state_index] = 1.0

#     distributions = np.zeros((n_steps + 1, env.n_states), dtype=float)

#     start_index = env.state_to_index[start_state]
#     distributions[0, start_index] = 1.0

#     for step in range(n_steps):
#         distributions[step + 1] = (distributions[step] @ policy_kernel)

#     return distributions

#Wrapper to keep Phase 1 backward compatability
def get_exact_step_distributions(env, kernel, Q, goal_index, goal_state, n_steps, start_state=None):
    policy = greedy_policy_from_q(Q,goal_index)
    return get_exact_policy_distributions(env, kernel, policy, goal_state=goal_state, n_steps=n_steps, start_state=start_state)


def dense_distributions_to_dicts(env, distributions, tolerance=1e-15):
  return [{state: float(probability) for state, probability in zip(env.states, distribution) if probability > tolerance} for distribution in distributions]

def greedy_policy_from_q(Q, goal_index):
    return np.argmax(Q[goal_index], axis=-1).astype(int)

#Gemini supported refactor to allow both deterministic and stochastic policy extraction
def build_policy_kernel(kernel, policy, goal_state_index=None):
    """
    Construct K_pi[s, s'] from either:

    - deterministic policy shaped [state], or
    - stochastic policy shaped [state, action].
    """
    kernel = np.asarray(kernel, dtype=float)
    policy = np.asarray(policy)

    n_states, n_actions, _ = kernel.shape

    if policy.ndim == 1:
        if policy.shape != (n_states,):
            raise ValueError(
                "Deterministic policy must have shape "
                f"({n_states},)"
            )

        policy_kernel = kernel[
            np.arange(n_states),
            policy.astype(int),
        ].copy()

    elif policy.ndim == 2:
        if policy.shape != (n_states, n_actions):
            raise ValueError(
                "Stochastic policy must have shape "
                f"({n_states}, {n_actions})"
            )

        policy_kernel = np.einsum(
            "sa,san->sn",
            policy,
            kernel,
        )

    else:
        raise ValueError(
            "Policy must contain selected actions or "
            "state-action probabilities"
        )

    if goal_state_index is not None:
        policy_kernel[goal_state_index] = 0.0
        policy_kernel[
            goal_state_index,
            goal_state_index,
        ] = 1.0

    return policy_kernel


def get_exact_policy_distributions(env, kernel, policy, goal_state, n_steps, start_state=None):
  if start_state is None:
    start_state = env.start_state

  goal_index = env.state_to_index[goal_state]
  start_index = env.state_to_index[start_state]

  policy_kernel = build_policy_kernel(kernel, policy, goal_state_index=goal_index)

  distributions = np.zeros((n_steps + 1, env.n_states), dtype=float)
  distributions[0, start_index] = 1.0
  for step in range(n_steps):
    distributions[step + 1] = (distributions[step] @ policy_kernel)

  return distributions