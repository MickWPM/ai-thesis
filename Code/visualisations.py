import matplotlib.pyplot as plt
import numpy as np


def make_probability_grid(env, distribution):
  grid = np.full((env.dimensions, env.dimensions),np.nan)

  for state in env.states:
    x, y = state
    grid[y, x] = 0.0

  for state, probability in distribution.items():
    x, y = state
    grid[y, x] = probability

  return grid


def draw_distribution(ax, env, distribution, step, title, goal_state, live_trajectory=None):
  grid = make_probability_grid(env,distribution)

  colour_map = plt.cm.viridis.copy()
  colour_map.set_bad(color="dimgray")

  image = ax.imshow(grid, cmap=colour_map, origin="lower", vmin=0.0, vmax=1.0)

  for state, probability in distribution.items():
    if probability > 0:
      x, y = state

      text_colour = ("white" if probability < 0.5 else "black")

      ax.text(x, y, f"{probability:.2f}", ha="center",va="center", color=text_colour,fontsize=8)

  start_x, start_y = env.start_state
  ax.text(start_x, start_y, "S", ha="left",va="bottom", color="red", fontweight="bold")

  goal_x, goal_y = goal_state
  ax.text(goal_x, goal_y, "G", ha="left", va="bottom",color="red",fontweight="bold")


  if live_trajectory is not None:
    path = live_trajectory[:min(step + 1, len(live_trajectory))]
    path_x = [state[0] for state in path]
    path_y = [state[1] for state in path]

    ax.plot(
        path_x,
        path_y,
        color="red",
        linewidth=2,
        alpha=0.8,
    )

    ax.scatter(
        path_x[-1],
        path_y[-1],
        color="red",
        edgecolor="white",
        s=100,
        zorder=3,
        label="Live agent",
    )

    ax.legend(loc="upper right")

  ax.set_title(title)
  ax.set_xlabel("X coordinate")
  ax.set_ylabel("Y coordinate")

  ax.set_xticks(np.arange(env.dimensions))
  ax.set_yticks(np.arange(env.dimensions))

  return image

def show_distribution(env, step_distributions, step, goal_state, live_trajectory=None):
  fig, ax = plt.subplots(figsize=(8, 8))

  image = draw_distribution(
      ax,
      env,
      step_distributions[step],
      step,
      title=f"P_hat probabilities at step {step}",
      goal_state=goal_state,
      live_trajectory=live_trajectory,
  )

  fig.colorbar(
      image,
      ax=ax,
      label="Probability",
      shrink=0.8,
  )

  plt.show()

def show_distribution_comparison(env, model_distributions, environment_distributions, step, goal_state, live_trajectory=None):
  fig, axes = plt.subplots(1, 2, figsize=(16, 7))

  image = draw_distribution(
      axes[0],
      env,
      model_distributions[step],
      step,
      title=f"P_hat distribution — step {step}",
      goal_state=goal_state,
      live_trajectory=live_trajectory,
  )

  draw_distribution(
      axes[1],
      env,
      environment_distributions[step],
      step,
      title=f"Environment distribution — step {step}",
      goal_state=goal_state,
      live_trajectory=live_trajectory,
  )

  fig.colorbar(
      image,
      ax=axes,
      label="Probability",
      shrink=0.8,
  )

  plt.show()