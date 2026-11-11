"""Plot the generated date comparison and matched deadline sensitivity."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LABELS = {"marginal": "Marginal score", "native_strict": "Durner strict",
          "nominal": "Nominal", "rollout": "Continuation rollout"}
COLORS = {"marginal": "#697481", "native_strict": "#A56538",
          "nominal": "#164A60", "rollout": "#587C49"}


def plot_budget(input_suffix="poll_history_results", output_stem="route_id_budget_poll_history"):
    budgets = (128, 512)
    summaries = [next(iter(json.loads((HERE / f"budget_{budget}_{input_suffix}.json").read_text())
                           ["summary"].values()))["metrics"] for budget in budgets]
    labels = {"full_rollout": "Full rollout", "fixed_h_rollout": "Fixed H",
              "material_h_rollout": "Smaller H", "belief_uct": "Belief UCT"}
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.15), sharey=True)
    for ax, metric, scale, xlabel in zip(axes, ("planning_transitions", "total_ms"), (1., 1000.),
                                         ("Mean transitions per journey", "Mean seconds per journey")):
        for (method, label), color, marker in zip(labels.items(), COLORS.values(), ("o", "s", "^", "D")):
            x = [summary[method][metric] / scale for summary in summaries]
            y = [summary[method]["loss"] for summary in summaries]
            ax.plot(x, y, marker=marker, markersize=4, linewidth=1.2, color=color, label=label)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Mean capped loss (min)")
    axes[1].legend(fontsize=8, frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(HERE / f"{output_stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    chronological = json.loads((HERE / "chronological_results.json").read_text())
    deadlines = json.loads((HERE / "deadline_results.json").read_text())
    dated = defaultdict(lambda: defaultdict(list))
    for case in chronological["cases"]:
        for row in case["rows"]:
            dated[row["method"]][case["spec"][0]].append(row["loss"])
    dates = sorted({case["spec"][0] for case in chronological["cases"]})
    caps = sorted({case["spec"][3] for case in deadlines["cases"]})
    figure, (daily, sensitivity) = plt.subplots(1, 2, figsize=(9.1, 3.2),
                                                 gridspec_kw={"width_ratios": [1.8, 1]})
    for method in chronological["methods"]:
        daily.plot(range(len(dates)), [mean(dated[method][date]) for date in dates],
                   color=COLORS[method], label=LABELS[method], linewidth=1.3)
        means = [deadlines["summary"][f"stress=True/cap={cap}"]["metrics"][method]["arrived"]
                 for cap in caps]
        sensitivity.plot(caps, means, color=COLORS[method], marker="o", markersize=4, linewidth=1.4)
    ticks = list(range(0, len(dates), 7))
    daily.set_xticks(ticks, [dates[i][5:] for i in ticks])
    daily.set(xlabel="Calibration date", ylabel="Mean capped loss (min)")
    daily.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    sensitivity.set_xticks(caps)
    sensitivity.set(xlabel="Deadline (min)", ylabel="Arrival fraction", ylim=(0, 1.06))
    for axes in (daily, sensitivity):
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", alpha=.22)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(HERE / f"chronological_panel.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    plot_budget()


if __name__ == "__main__":
    main()
