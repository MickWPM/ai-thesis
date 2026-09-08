from bellman_inverter import BellmanInverterStochastic
from goal_conditioned_training import make_goal_rewards,  make_physical_goal_rewards, train_goal_conditioned_agent
import time

def get_inverted_model(environment, goals=None, starts=None, 
                       gamma=0.99, reward_perturbation=0.01,
                       n_episodes=50_000, max_steps=200,
                       seed_rewards=32, seed_train=8):
    if goals is None:
        goals = list(environment.states)
    else:
        goals = list(goals)
    supports = environment.make_local_supports()
    inverter = BellmanInverterStochastic(gamma=gamma)
    rewards, done = make_goal_rewards(
      environment,
      goals,
      perturbation=reward_perturbation,
      seed=seed_rewards,
    )

    Q, visits, success_history = (
      train_goal_conditioned_agent(
          environment,
          rewards,
          done,
          goals=goals,
          starts=starts,
          n_episodes=n_episodes,
          max_steps=max_steps,
          gamma=gamma,
          seed=seed_train,
        )
    )

    P_hat = inverter.fit(
      Q,
      rewards,
      done,
      supports,
    )

    M = inverter.M
    bellman_residual = inverter.bellman_residual
    return Q, P_hat, bellman_residual

def get_seeded_inverted_models(environment, goals=None, starts=None, training_seeds=(8, 18, 28, 38, 48), gamma=0.99, reward_perturbation=0.01, n_episodes=100_000, max_steps=200, seed_rewards=32):
  resolved_goals = (list(environment.states) if goals is None else list(goals))
  results = []
  for training_seed in training_seeds:
    print(f"Training seed {training_seed} with {len(resolved_goals)} goals")

    Q, P_hat, bellman_residual = get_inverted_model(
        environment,
        goals=resolved_goals,
        starts=starts,
        gamma=gamma,
        reward_perturbation=reward_perturbation,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed_rewards=seed_rewards,
        seed_train=training_seed
    )

    results.append({
        "training_seed": training_seed,
        "goals": resolved_goals,
        "Q": Q,
        "P_hat": P_hat,
        "bellman_residual": bellman_residual
    })

  return results


def train_phase2_agents(env, definitions, training_seeds, episodes_per_goal, gamma, perturbation, reward_seed, max_steps, TRAINING_HISTORY_WINDOW=100):
  runs = []
  for training_seed in training_seeds:
    for name, definition in definitions.items():
      goals = list(definition["goals"])
      rewards, done = make_physical_goal_rewards(env, goals, perturbation=perturbation, seed=reward_seed)
      started = time.perf_counter()
      Q, visits, success_history = train_goal_conditioned_agent(env, rewards, done, goals=goals, starts=definition["starts"], n_episodes=episodes_per_goal * len(goals), max_steps=max_steps, gamma=gamma, seed=training_seed)
      runs.append({
                "training seed": training_seed,
                "name": name,
                "room": definition["room"],
                "goals": goals,
                "starts": list(definition["starts"]),
                "Q": Q,
                "visits": visits,
                "success_history": success_history,
                "rewards": rewards,
                "done": done,
                "training seconds": time.perf_counter() - started,
            })
      print(f"Seed {training_seed}: trained {name} ({success_history[-TRAINING_HISTORY_WINDOW:].mean():.1%} recent success)")

  return runs


def extract_world_models(env, runs, gamma):
  supports = env.make_local_supports()
  extracted = []
  for run in runs:
    inverter = BellmanInverterStochastic(gamma=gamma)
    started = time.perf_counter()
    P_hat = inverter.fit(run["Q"], run["rewards"], run["done"], supports)
    extracted_run = dict(run)
    extracted_run.update({
            "P_hat": P_hat,
            "bellman_residual": inverter.bellman_residual,
            "masked_bellman_residual": inverter.masked_bellman_residual,
            "extraction seconds": time.perf_counter() - started,
        })
    extracted.append(extracted_run)
    print(f"Seed {run['training seed']}: extracted {run['name']}")
  return extracted