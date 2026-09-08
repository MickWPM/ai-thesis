import numpy as np
import pandas as pd
from scipy.stats import entropy
from inverse_prediction import get_exact_step_distributions, build_environment_kernel, get_exact_policy_distributions, greedy_policy_from_q

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


#Refactored backwards compatable function - updates to original intgrated from Gemini proposed refactor
def compare_policy_kernels_exact(env, policy, reference_kernel, candidate_kernel, start_state, goal_state, n_steps, kl_epsilon=1e-12):
    reference = get_exact_policy_distributions(env, reference_kernel, policy, goal_state, n_steps, start_state)
    candidate = get_exact_policy_distributions(env, candidate_kernel, policy, goal_state, n_steps, start_state)

    tv = (0.5 * np.abs(reference - candidate).sum(axis=1))

    reference_kl = np.clip(reference, kl_epsilon, None)
    candidate_kl = np.clip(candidate, kl_epsilon, None)
    reference_kl /= reference_kl.sum(axis=1, keepdims=True)
    candidate_kl /= candidate_kl.sum(axis=1, keepdims=True)
    kl = np.sum(reference_kl * (np.log(reference_kl) - np.log(candidate_kl)), axis=1)

    goal_index = env.state_to_index[goal_state]
    reference_goal = reference[:, goal_index]
    candidate_goal = candidate[:, goal_index]

    return {
        "reference distributions": reference,
        "candidate distributions": candidate,
        "trajectory TV": tv,
        "trajectory KL": kl,
        "reference goal curve": reference_goal,
        "candidate goal curve": candidate_goal,
        "goal error": np.abs(reference_goal - candidate_goal),
        "goal bias": (candidate_goal - reference_goal)}

def compare_exact_distributions(reference, candidate, goal_index, epsilon=1e-12):
  tv = 0.5 * np.abs(reference - candidate).sum(axis=1)

  reference_kl = np.clip(reference, epsilon, None)
  candidate_kl = np.clip(candidate, epsilon, None)
  reference_kl /= reference_kl.sum(axis=1, keepdims=True)
  candidate_kl /= candidate_kl.sum(axis=1, keepdims=True)
  kl = np.sum(reference_kl * np.log(reference_kl / candidate_kl), axis=1)

  reference_goal = reference[:, goal_index]
  candidate_goal = candidate[:, goal_index]
  goal_error = np.abs(reference_goal - candidate_goal)
  goal_bias = candidate_goal - reference_goal

  return {
        "trajectory TV": tv[1:].mean(),
        "maximum trajectory TV": tv[1:].max(),
        "final trajectory TV": tv[-1],
        "trajectory KL": kl[1:].mean(),
        "goal-curve MAE": goal_error[1:].mean(),
        "final goal error": goal_error[-1],
        "final goal bias": goal_bias[-1],
    }

def qualify_trained_policies(env, runs, true_kernel, n_steps, POLICY_SUCCESS_THRESHOLD):
  rows = []
  for run in runs:
    for goal_index, goal_state in enumerate(run["goals"]):
      policy = greedy_policy_from_q(run["Q"], goal_index)
      state_goal_index = env.state_to_index[goal_state]
      for start_state in run["starts"]:
        distributions = get_exact_policy_distributions(env, true_kernel, policy, goal_state, n_steps, start_state)
        curve = distributions[:, state_goal_index]
        rows.append({
                "training seed": run["training seed"],
                "agent": run["name"],
                "goal": goal_state,
                "start": start_state,
                "final success": curve[-1],
                "goal-curve AUC": curve[1:].mean(),
                "mean capped steps": np.sum(1.0 - curve[:-1]),
                "qualified": curve[-1] >= POLICY_SUCCESS_THRESHOLD,
            })
  return pd.DataFrame(rows)


def evaluate_cross_models(env, runs, true_kernel, n_steps):
  rows = []

  for training_seed in sorted({run["training seed"] for run in runs}):
    seeded_runs = [run for run in runs if run["training seed"] == training_seed]

    for source in seeded_runs:
      for goal_index, goal_state in enumerate(source["goals"]):
        policy = greedy_policy_from_q(source["Q"], goal_index)
        state_goal_index = env.state_to_index[goal_state]

        for start_state in source["starts"]:
          own_reference = get_exact_policy_distributions(env, source["P_hat"], policy, goal_state, n_steps, start_state)
          true_reference = get_exact_policy_distributions(env, true_kernel, policy, goal_state, n_steps, start_state)

          for candidate in seeded_runs:
            candidate_distribution = get_exact_policy_distributions(env, candidate["P_hat"], policy, goal_state, n_steps, start_state)
            internal = compare_exact_distributions(own_reference, candidate_distribution, state_goal_index)
            external = compare_exact_distributions(true_reference, candidate_distribution, state_goal_index)
            row = {
                "training seed": training_seed,
                "candidate model": candidate["name"],
                "source policy": source["name"],
                "goal": goal_state,
                "start": start_state,
            }
            row.update({f"internal {key}": value for key, value in internal.items()})
            row.update({f"true {key}": value for key, value in external.items()})
            rows.append(row)

    print(f"Completed cross-model assessment for seed {training_seed}")

  return pd.DataFrame(rows)

def cross_matrix(results, metric, AGENT_NAMES):
  return results.groupby(["candidate model", "source policy"])[metric].mean().unstack().reindex(index=AGENT_NAMES, columns=AGENT_NAMES)
