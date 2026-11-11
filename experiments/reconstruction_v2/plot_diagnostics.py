"""Three reproducible diagnostic figures from exact finite-bank results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.titlesize": 11, "legend.frameon": False})


def save(fig, name):
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    transit = json.loads((HERE / "exact_transit_results.json").read_text())["cases"]
    mechanism = json.loads((HERE / "mechanism_results.json").read_text())["rows"]
    rows = [row for row in mechanism if row["timeout"] == 120 and row["hub_menu"] == ["A"]]

    # Conclusion: a finite-subnet quality/time trade-off can be measured without
    # confusing this reference implementation's latency with a city bottleneck.
    fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
    for i, case in enumerate(transit):
        full, restricted = case["full_causal_optimum"], case["restricted_causal_optimum"]
        ax.plot([restricted["solve_ms"], full["solve_ms"]],
                [restricted["value"], full["value"]], color="#B5B5B5", linewidth=1)
        ax.scatter(restricted["solve_ms"], restricted["value"], color=ORANGE, marker="s", s=35,
                   label="Restricted optimum" if i == 0 else None)
        ax.scatter(full["solve_ms"], full["value"], color=BLUE, s=35,
                   label="Full causal optimum" if i == 0 else None)
        ax.annotate(f'OD {i // 2 + 1}, cap {case["timeout_cap"]}',
                    (restricted["solve_ms"], restricted["value"]),
                    xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", xlabel="Reference solve time (ms; allocation tracing enabled)",
           ylabel="Expected capped loss (min)", title="Exact quality and computation on two transit subnets")
    ax.legend(loc="lower left")
    ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
    save(fig, "quality_computation")

    # Conclusion: the same total loss has a different attribution when H changes.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), sharey=True, layout="constrained")
    labels = ["Opposite\nearly", "Independent\nearly", "Together\nearly", "Opposite\nlate"]
    for ax, menu, title in zip(axes, (["A"], ["A", "B"]), ("Hub menu: A only", "Hub menu: A and B")):
        selected = [r for r in mechanism if r["timeout"] == 120 and r["hub_menu"] == menu]
        support = np.array([r["support_loss"] for r in selected])
        within = np.array([r["within_candidate_loss"] for r in selected])
        ax.bar(range(4), support, color=ORANGE, width=.65, label="Candidate restriction")
        ax.bar(range(4), within, bottom=support, color=BLUE, width=.65, label="Within-candidate decisions")
        ax.set_xticks(range(4), labels)
        ax.set_title(title)
        ax.set_ylim(0, 16)
        ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Policy loss above full optimum (min)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Exact loss decomposition: a marginal policy choosing the 25-minute direct trip", fontsize=10)
    save(fig, "loss_decomposition")

    # Conclusion: identical one-line marginals do not identify option value.
    fig, ax = plt.subplots(figsize=(6.2, 3.7), layout="constrained")
    x = np.arange(4)
    ax.bar(x - .18, [r["full_optimum"] for r in rows], .36, color=BLUE, label="Full causal optimum")
    ax.bar(x + .18, [r["restricted_optimum"] for r in rows], .36, color=ORANGE, label="Hub menu: A only")
    for i, row in enumerate(rows):
        ax.text(i, 26, f'Gap {row["support_loss"]:g}', ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Expected journey cost (min)", ylim=(0, 33),
           title="Joint disruption and message timing change the value of a backup")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, "correlation_information_timing")
    print("Wrote three figures as PNG, SVG and PDF")


if __name__ == "__main__":
    main()
