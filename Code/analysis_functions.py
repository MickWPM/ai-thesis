import numpy as np
from scipy.stats import entropy

def state_distribution_to_environment_distribution(state_distribution, env):
  distribution = np.zeros(env.n_states)
  for state, prob in state_distribution.items():
    distribution[env.state_to_index[state]] = prob
  distribution /= distribution.sum()
  return distribution


def get_kl_from_reference(env, reference_distributions, comparison_distributions,
                          epsilon=1e-10):
  kl = []
  n_steps = min(len(reference_distributions), len(comparison_distributions))
  for i in range(1, n_steps):
    reference = state_distribution_to_environment_distribution(reference_distributions[i], env)
    comparison =  state_distribution_to_environment_distribution(comparison_distributions[i], env)
    reference = np.clip(reference, epsilon, None)
    comparison = np.clip(comparison, epsilon, None)
    reference /= reference.sum()
    comparison /= comparison.sum()
    kl.append(entropy(reference, comparison))
  return np.array(kl)

def get_tv_from_reference(env, reference_distributions, comparison_distributions):
  tv = []
  n_steps = min(len(reference_distributions), len(comparison_distributions))
  for i in range(1, n_steps):
    reference = state_distribution_to_environment_distribution(reference_distributions[i], env)
    comparison =  state_distribution_to_environment_distribution(comparison_distributions[i], env)
    reference /= reference.sum()
    comparison /= comparison.sum()
    tv.append(0.5 * np.abs(reference - comparison).sum())
  return np.array(tv)