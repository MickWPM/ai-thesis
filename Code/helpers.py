from bellman_inverter import BellmanInverterStochastic
from goal_conditioned_training import make_goal_rewards, train_goal_conditioned_agent

def get_inverted_model(environment, goals=None, starts=None, 
                       gamma=0.99, reward_perturbation=0.01,
                       n_episodes=50_000, max_steps=200):
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
      seed=32,
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
          seed=8,
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