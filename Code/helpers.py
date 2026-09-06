from bellman_inverter import BellmanInverterStochastic
from goal_conditioned_training import make_goal_rewards, train_goal_conditioned_agent

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