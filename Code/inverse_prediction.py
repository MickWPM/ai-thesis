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