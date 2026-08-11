import numpy as np
from scipy.optimize import linprog

class BellmanInverterDeterministic:
  def __init__(self, gamma=0.99):
    self.gamma = gamma

  def fit(self, Q, rewards, done):
    Q = np.asarray(Q, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    done = np.asarray(done, dtype=bool)

    n_goals, n_states, n_actions = Q.shape

    V = Q.max(axis=-1)

    #M(s′, g) := r(s′, g) + γV (s′, g). equation (1), p4
    M = rewards + self.gamma * (~done) * V
    P_hat = np.zeros( (n_states, n_actions, n_states))

    for state in range(n_states):
      valid_goals = ~done[:, state]
      for action in range(n_actions):
        q = Q[valid_goals, state, action][:, None]

        errors = np.sum(np.abs(M[valid_goals] - q), axis=0)
        next_state = np.argmin(errors)
        P_hat[state, action, next_state] = 1.0

    self.M = M
    self.P = P_hat

    #MPφ(s, a) = Q(s, a) equation (2) p4
    reconstructed_Q = np.einsum(
        "san,gn->gsa",
        P_hat,
        M,
    )
    self.bellman_residual = reconstructed_Q - Q

    return P_hat

class BellmanInverterStochastic:
  def __init__(self, gamma=0.99):
    self.gamma = gamma

  def fit(self, Q, rewards, done, supports):
    Q = np.asarray(Q, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    done = np.asarray(done, dtype=bool)

    n_goals, n_states, n_actions = Q.shape

    V = Q.max(axis=-1)

    #M(s′, g) := r(s′, g) + γV (s′, g). equation (1), p4
    M = rewards + self.gamma * (~done) * V
    P_hat = np.zeros( (n_states, n_actions, n_states))

    for state in range(n_states):
      support = np.asarray(supports[state], dtype=int)
      valid_goals = ~done[:, state]
      M_local = M[valid_goals][:, support]
      n_valid_goals = int(valid_goals.sum())

      n_candidates = len(support)

      for action in range(n_actions):
        q = Q[valid_goals, state, action]

        objective = np.concatenate([
            np.zeros(n_candidates),
            np.ones(n_valid_goals),
        ])

        identity = np.eye(n_valid_goals)

        A_ub = np.vstack([
            np.hstack([
                M_local,
                -identity,
            ]),
            np.hstack([
                -M_local,
                -identity,
            ]),
        ])

        b_ub = np.concatenate([
            q,
            -q,
        ])

        A_eq = np.concatenate([
            np.ones(n_candidates),
            np.zeros(n_valid_goals),
        ])[None, :]

        b_eq = [1.0]

        bounds = [
            (0.0, None)
        ] * len(objective)

        result = linprog(
            c=objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            raise RuntimeError(
                f"LP failed for state={state}, "
                f"action={action}: {result.message}"
            )

        # The first K variables are the recovered
        # transition probabilities.
        probabilities = result.x[:n_candidates]

        # Clean up tiny numerical errors.
        probabilities = np.clip(
            probabilities,
            0.0,
            None,
        )
        probabilities /= probabilities.sum()

        # Store probabilities in the full kernel.
        P_hat[
            state,
            action,
            support,
        ] = probabilities

    # Check how well P_hat reconstructs Q.
    reconstructed_Q = np.einsum(
        "san,gn->gsa",
        P_hat,
        M,
    )

    residual = reconstructed_Q - Q
    valid_entries = np.broadcast_to(
        (~done)[:, :, None],
        Q.shape,
    )
    self.bellman_residual = residual
    self.masked_bellman_residual = residual[
        valid_entries
    ]

    self.M = M
    self.P = P_hat
    return P_hat


def get_next_states(env, state, Q, P_hat,goal_index):
  state_index = env.state_to_index[state]

  action_index = int(np.argmax(Q[goal_index, state_index]))
  probabilities = P_hat[state_index,action_index]
  next_states = []

  for next_state, probability in zip(env.states,probabilities):
    if probability > 1e-10:
      next_states.append((next_state, probability))

  return next_states