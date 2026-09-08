import numpy as np

def compute_q_values(kernel, reward, value, gamma, goal_index):
  target = reward + gamma * value
  Q = kernel @ target   
  Q[goal_index] = 0.0 
  return Q

def value_iteration_for_goal(env, kernel, goal_state, gamma=0.99, tolerance=1e-10, max_iterations=20000):
  goal_index = env.state_to_index[goal_state]
  reward = np.zeros(env.n_states)
  reward[goal_index] = 1.0
  value = np.zeros(env.n_states)
  
  for iteration in range(max_iterations):
    Q = compute_q_values(kernel, reward, value, gamma, goal_index)
    updated_value = Q.max(axis=1)
    
    if np.max(np.abs(updated_value - value)) < tolerance:
      value = updated_value
      break    
    value = updated_value
  final_Q = compute_q_values(kernel, reward, value, gamma, goal_index)
  policy = np.argmax(final_Q, axis=1).astype(int)
  
  return policy, value, iteration + 1