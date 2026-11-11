"""Plot measured capped loss against elapsed time for the matched budget panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
STYLES = {"full_rollout": ("Full menu rollout", "#0072B2", "o", (0, -14)),
          "fixed_rollout": ("Fixed H, all initial labels", "#D55E00", "s", (0, -14)),
          "material_h_rollout": ("Fixed H, top five per stop", "#009E73", "D", (0, 8)),
          "full_uct": ("Full menu belief UCT", "#6A3D9A", "^", (0, 8))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "planning_budget_results.json")
    parser.add_argument("--name", default="committed_budget_curve")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    budgets = data["config"]["budgets"]
    rows = data["cases"]
    paired_cases = {(row["panel"], row["index"]) for row in rows}
    if data["execution_errors"] or any(
            data["summary"]["all"][str(budget)]["complete_worlds"] != len(paired_cases)
            for budget in budgets):
        raise ValueError("The figure requires the same complete worlds at each budget")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    fig.subplots_adjust(left=.13, right=.97, top=.86, bottom=.31)
    for method in data["config"]["methods"]:
        label, color, marker, offset = STYLES[method]
        groups = [data["summary"]["all"][str(budget)]["methods"][method] for budget in budgets]
        seconds = [group["median_total_ms"] / 1000 for group in groups]
        losses = [group["mean_loss"] for group in groups]
        ax.plot(seconds, losses, color=color, marker=marker, markersize=5, linewidth=1.4, label=label)
        for x, y, budget in zip(seconds, losses, budgets):
            ax.annotate(str(budget), (x, y), xytext=offset, textcoords="offset points",
                        ha="center", fontsize=8, color=color)
    reference = {(row["panel"], row["index"]): row for row in rows}
    ax.axhline(mean(row["nominal_loss"] for row in reference.values()),
               color="#666666", linestyle="--", linewidth=1, label="Online nominal cost")
    ax.axhline(mean(row["v2_rollout_loss"] for row in reference.values()),
               color="#999999", linestyle=":", linewidth=1.2, label="V2 rollout cost")
    ax.set(xscale="log", xlabel="Median end-to-end time per journey (s)",
           ylabel="Mean capped journey loss (min)")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 4, 6, 8)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
    ax.set_axisbelow(True)
    ax.margins(x=.10, y=.18)
    ax.legend(loc="upper left", ncols=2, fontsize=8, frameon=True,
              facecolor="white", edgecolor="none", framealpha=1)
    fig.text(.13, .94, f"Planning quality and time on {len(paired_cases)} frozen generated worlds",
             fontsize=11, fontweight="bold", ha="left")
    fig.text(.13, .15, "Point labels give the transition cap per decision. Actual work may fall below the cap.",
             fontsize=8, ha="left")
    fig.text(.13, .102, "Menus H are fixed and trip-closed. The causal nominal tail obeys each menu.",
             fontsize=8, ha="left")
    fig.text(.13, .054, f"One {data['config']['workers']}-worker batch. Menu loss gaps are empirical. Reference lines show costs only.",
             fontsize=8, ha="left")
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{args.name}.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
