from __future__ import annotations

import csv
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STYLE_DIR = Path.home() / ".codex" / "skills" / "autoresearch-skill" / "scripts"
sys.path.insert(0, str(STYLE_DIR))

import matplotlib.pyplot as plt

from style_presets import OKABE_ITO, rcparams


def main() -> None:
    rows = list(csv.DictReader((HERE / "autoresearch-results.tsv").open(encoding="utf-8"), delimiter="\t"))
    iterations = [int(row["iteration"]) for row in rows]
    values = [float(row["metric_value"]) for row in rows]
    statuses = [row["status"] for row in rows]
    best = []
    running = float("-inf")
    for value in values:
        running = max(running, value)
        best.append(running)

    rcparams()
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(iterations, best, color=OKABE_ITO[4], label="Best completion")
    for iteration, value, status in zip(iterations, values, statuses):
        filled = status in {"baseline", "kept", "reference"}
        axis.scatter(
            [iteration],
            [value],
            s=48,
            facecolor=OKABE_ITO[1] if filled else "white",
            edgecolor=OKABE_ITO[4],
            linewidth=1.2,
            zorder=3,
        )
    axis.axhline(1.0, color=OKABE_ITO[5], linestyle="--", label="Target")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Paper-suite completion fraction")
    axis.set_xlim(left=-0.2)
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    fig.savefig(HERE / "progress.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
