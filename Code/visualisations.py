#Large portions of this script refactored or developed with Gemini support for aesthetics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from inverse_prediction import greedy_policy_from_q, get_exact_policy_distributions


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

def plot_training_setup(env, agent_definitions, held_out_goals):
    room_colours = {
        "south-west": "#a8ddb5",
        "north-west": "#b3cde3",
        "south-east": "#fdd0a2",
        "north-east": "#d4b9da",
        "doorway": "#d9d9d9",
    }
    fig, axes = plt.subplots(1, len(agent_definitions), figsize=(20, 4.3), sharex=True, sharey=True)

    for ax, (name, definition) in zip(axes, agent_definitions.items()):
        for state in env.states:
            ax.scatter(*state, marker="s", s=85, color=room_colours[env._get_room(state)], alpha=0.75)

        goals = np.asarray(definition["goals"])
        starts = np.asarray(definition["starts"])
        held_out = np.asarray(held_out_goals)
        ax.scatter(goals[:, 0], goals[:, 1], color="tab:red", s=55, zorder=3)
        ax.scatter(starts[:, 0], starts[:, 1], color="black", marker="x", s=70, linewidth=2, zorder=4)
        ax.scatter(held_out[:, 0], held_out[:, 1], color="dimgray", marker="*", s=65, zorder=2)
        ax.set_title(name)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks(range(11))
        ax.set_yticks(range(11))
        ax.set_aspect("equal")

    handles = [
        Line2D([], [], marker="o", linestyle="", color="tab:red", label="training goal"),
        Line2D([], [], marker="x", linestyle="", color="black", label="training start"),
        Line2D([], [], marker="*", linestyle="", color="dimgray", label="held-out goal (Phase 3 only)"),
    ]
    fig.suptitle("Phase 2 training distributions", fontsize=15)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0.10, 1, 0.92))
    plt.show()

def plot_visit_maps(env, runs, agents_to_be_trained):
    visit_maps = {}
    maximum = 0.0
    agent_names = list(agents_to_be_trained)
    for name in agent_names:
        state_visits = sum(
            run["visits"].sum(axis=(0, 2))
            for run in runs if run["name"] == name
        )
        visit_maps[name] = np.log1p(state_visits)
        maximum = max(maximum, visit_maps[name].max())

    fig, axes = plt.subplots(1, len(agent_names), figsize=(20, 4.3), sharex=True, sharey=True)
    image = None

    for ax, name in zip(axes, agent_names):
        grid = np.full((env.dimensions, env.dimensions), np.nan)
        for state, value in zip(env.states, visit_maps[name]):
            grid[state[1], state[0]] = value
        image = ax.imshow(grid, origin="lower", cmap="magma", vmin=0.0, vmax=maximum)
        definition = agents_to_be_trained[name]
        goals = np.asarray(definition["goals"])
        starts = np.asarray(definition["starts"])
        ax.scatter(goals[:, 0], goals[:, 1], facecolors="none", edgecolors="cyan", s=60)
        ax.scatter(starts[:, 0], starts[:, 1], color="white", marker="x", s=50)
        ax.set_title(name)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")

    fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.05, pad=0.13, label="log(1 + state visits)")
    fig.suptitle("Training visitation across seeds", fontsize=15)
    fig.subplots_adjust(bottom=0.24, top=0.83, wspace=0.16)
    plt.show()

def show_phase2_trajectory_demo(env, runs, true_kernel, source_agent, goal_state, start_state, training_seed, steps=(1, 5, 10, 20), candidate_model=None):
  source = next(run for run in runs if run["training seed"] == training_seed and run["name"] == source_agent)
  candidate_name = source_agent if candidate_model is None else candidate_model
  candidate = next(run for run in runs if run["training seed"] == training_seed and run["name"] == candidate_name)
  goal_index = source["goals"].index(goal_state)
  policy = greedy_policy_from_q(source["Q"], goal_index)
  true_distributions = get_exact_policy_distributions(env, true_kernel, policy, goal_state, max(steps), start_state)
  model_distributions = get_exact_policy_distributions(env, candidate["P_hat"], policy, goal_state, max(steps), start_state)

  fig, axes = plt.subplots(2, len(steps), figsize=(4.2 * len(steps), 8.2), sharex=True, sharey=True)
  image = None

  for column, step in enumerate(steps):
    for row, (label, distributions) in enumerate([("True kernel", true_distributions), (f"{candidate_name} model", model_distributions)]):
      grid = np.full((env.dimensions, env.dimensions), np.nan)
      for state, probability in zip(env.states, distributions[step]):
        grid[state[1], state[0]] = probability

      image = axes[row, column].imshow(grid, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
      axes[row, column].scatter(*start_state, marker="x", color="red", s=75, linewidth=2)
      axes[row, column].scatter(*goal_state, marker="*", color="red", s=110)
      axes[row, column].set_title(f"{label} — step {step}")
      axes[row, column].set_xlim(0, 10)
      axes[row, column].set_ylim(0, 10)
      axes[row, column].set_aspect("equal")

  fig.suptitle(
      f"{source_agent}: start {start_state}, goal {goal_state}", fontsize=16
  )
  fig.colorbar(
      image, ax=axes, orientation="horizontal", fraction=0.04,
      pad=0.09, label="State probability"
  )
  fig.subplots_adjust(bottom=0.17, top=0.88, hspace=0.28, wspace=0.16)
  plt.show()