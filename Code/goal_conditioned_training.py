import random
import numpy as np

def make_goal_rewards(env, goals, perturbation=0.01,seed=32):
  rng = np.random.default_rng(seed)
  n_goals = len(goals)

  rewards = -rng.uniform(0.0, perturbation,size=(n_goals, env.n_states))

  done = np.zeros((n_goals, env.n_states),dtype=bool)

  for goal_index, goal_state in enumerate(goals):
    state_index = env.state_to_index[goal_state]
    rewards[goal_index, state_index] += 1.0
    done[goal_index, state_index] = True

  return rewards, done


#Q shape is [goal, state, action]
def train_goal_conditioned_agent(env, rewards, done, 
                                 goals = None, starts=None, 
                                 n_episodes=50_000, max_steps=200, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, learning_rate_exponent=0.6,
                                 seed=32):
  rng = np.random.default_rng(seed)
  random.seed(seed)

  if goals is None:
    goals = list(env.states)
  else:
    goals = list(goals)

  if starts is None:
    starts = list(env.states)
  else:
    starts = list(starts)

  n_goals = len(goals)
  valid_starts = [
      [
          state
          for state in starts
          if state != goal
      ]
      for goal in goals
  ]

  Q = np.zeros((n_goals, env.n_states, env.n_actions))

  visits = np.zeros_like(Q, dtype=int)
  success_history = np.zeros(n_episodes, dtype=bool)

  for episode in range(n_episodes):
    progress = episode / max(n_episodes - 1, 1)
    epsilon = (epsilon_start + progress * (epsilon_end - epsilon_start))

    goal_index = int(rng.integers(n_goals))
    goal_state = goals[goal_index]

    possible_starts = (
        valid_starts[
            goal_index
        ]
    )

    if len(possible_starts) == 0:
      raise ValueError(f"No valid start exists for goal {goal_state}")


    start_number = int(rng.integers(len(possible_starts)))
    state = possible_starts[start_number]


    for step_number in range(max_steps):
      state_index = env.state_to_index[state]
      if rng.random() < epsilon:
        action = int(rng.integers(env.n_actions))
      else:
        action = int(np.argmax(Q[goal_index, state_index]))

      next_state = env.move(state,action)
      next_index = env.state_to_index[next_state]

      reward = float(rewards[goal_index, next_index])
      terminated = bool(done[goal_index, next_index])

      #Learning rate scale off visits.
      visits[goal_index, state_index,action] += 1
      learning_rate = 1.0 / (visits[goal_index, state_index, action] ** learning_rate_exponent)

      if terminated:
        target = reward
      else:
        target = (reward + gamma * np.max(Q[goal_index, next_index]))

      Q[goal_index, state_index, action] += learning_rate * (target - Q[goal_index, state_index,action])
      state = next_state

      if terminated:
        success_history[episode] = True
        break

  return Q, visits, success_history