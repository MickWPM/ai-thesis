import numpy as np
import pandas as pd
from scipy.stats import entropy
from inverse_prediction import get_exact_step_distributions, build_environment_kernel

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

def evaluate_seeded_models_exact(env, seeded_models, goals, goal_index, n_steps, start_state=None, kl_epsilon=1e-12):
  if start_state is None:
    start_state = env.start_state
  goal_state = goals[goal_index]
  goal_state_index = env.state_to_index[goal_state]
  P_true = build_environment_kernel(env)
  metric_rows = []
  summary_rows = []
  distributions_by_seed = {}

  for result in seeded_models:
    seed = result["training_seed"]
    Q = result["Q"]
    P_hat = result["P_hat"]

    true_exact = get_exact_step_distributions(env, kernel=P_true, Q=Q, goal_index=goal_index, goal_state=goal_state, n_steps=n_steps, start_state=start_state)
    model_exact = get_exact_step_distributions(env, kernel=P_hat, Q=Q, goal_index=goal_index, goal_state=goal_state, n_steps=n_steps, start_state=start_state)

    trajectory_tv = (0.5 * np.abs(true_exact - model_exact).sum(axis=1))
    trajectory_kl = np.sum(true_exact * (np.log(np.clip(true_exact, kl_epsilon, None)) - np.log(np.clip(model_exact, kl_epsilon, None))), axis=1)

    true_goal_curve = true_exact[:, goal_state_index]
    model_goal_curve = model_exact[:, goal_state_index]

    goal_error = np.abs(true_goal_curve - model_goal_curve)
    goal_bias = (model_goal_curve - true_goal_curve)

    transition_row_tv = (0.5 * np.abs(P_true - P_hat).sum(axis=-1))

    for step in range(n_steps + 1):
      metric_rows.append({
        "training seed": seed,
        "step": step,
        "trajectory TV": trajectory_tv[step],
        "trajectory KL": trajectory_kl[step],
        "true goal probability": (true_goal_curve[step]),
        "inferred goal probability": (model_goal_curve[step]),
        "goal probability error": goal_error[step],
        "goal probability bias": goal_bias[step]})

    summary_rows.append({
        "training seed": seed,
        "mean transition-row TV": (transition_row_tv.mean()),
        "maximum transition-row TV": (transition_row_tv.max()),
        "mean trajectory TV": (trajectory_tv[1:].mean()),
        "maximum trajectory TV": (trajectory_tv[1:].max()),
        "step of maximum TV": int(np.argmax(trajectory_tv[1:]) + 1),
        "final trajectory TV": trajectory_tv[-1],
        "mean trajectory KL": (trajectory_kl[1:].mean()),
        "goal-curve MAE": goal_error[1:].mean(),
        "final goal error": goal_error[-1],
        "final goal bias": goal_bias[-1]})

    distributions_by_seed[seed] = {"true": true_exact, "inferred": model_exact}

  return (pd.DataFrame(metric_rows),pd.DataFrame(summary_rows), distributions_by_seed, P_true)