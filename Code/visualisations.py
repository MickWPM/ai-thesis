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


#This function generated as a standalone visualisation helper using Gemini
def plot_seeded_phase1_metrics(
    metric_df,
    goal_state,
    start_state,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    def plot_metric_band(
        ax,
        column,
        title,
        ylabel,
        colour,
    ):
        values = metric_df.pivot(
            index="step",
            columns="training seed",
            values=column,
        )

        # Show every individual training seed faintly.
        for seed in values.columns:
            ax.plot(
                values.index,
                values[seed],
                color=colour,
                alpha=0.18,
                linewidth=1,
            )

        mean = values.mean(axis=1)
        minimum = values.min(axis=1)
        maximum = values.max(axis=1)

        ax.fill_between(
            values.index,
            minimum,
            maximum,
            color=colour,
            alpha=0.2,
            label="Seed min–max",
        )

        ax.plot(
            values.index,
            mean,
            color=colour,
            linewidth=2.5,
            label="Seed mean",
        )

        ax.set_title(title)
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        # Place the legend outside the graph.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            frameon=False,
        )

    plot_metric_band(
        axes[0],
        column="trajectory TV",
        title="Exact trajectory TV",
        ylabel="TV distance",
        colour="tab:blue",
    )

    axes[0].set_ylim(bottom=0.0, top=1.0)

    plot_metric_band(
        axes[1],
        column="trajectory KL",
        title="Exact trajectory KL",
        ylabel="KL divergence",
        colour="tab:orange",
    )

    true_values = metric_df.pivot(
        index="step",
        columns="training seed",
        values="true goal probability",
    )

    inferred_values = metric_df.pivot(
        index="step",
        columns="training seed",
        values="inferred goal probability",
    )

    for values, colour, label in [
        (true_values, "black", "True kernel"),
        (
            inferred_values,
            "tab:purple",
            "Inferred model",
        ),
    ]:
        mean = values.mean(axis=1)
        minimum = values.min(axis=1)
        maximum = values.max(axis=1)

        # Individual seed curves.
        for seed in values.columns:
            axes[2].plot(
                values.index,
                values[seed],
                color=colour,
                alpha=0.10,
                linewidth=1,
            )

        axes[2].fill_between(
            values.index,
            minimum,
            maximum,
            color=colour,
            alpha=0.15,
        )

        axes[2].plot(
            values.index,
            mean,
            color=colour,
            linewidth=2.5,
            label=f"{label} mean",
        )

    axes[2].set_title("Goal-reaching probability")
    axes[2].set_xlabel("Prediction step")
    axes[2].set_ylabel("Probability reached")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.3)

    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        f"Training-seed variation: "
        f"start {start_state}, goal {goal_state}",
        fontsize=15,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.94))
    plt.show()