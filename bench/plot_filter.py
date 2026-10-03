from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, NullFormatter

from plot_recall_qps import THEMES

STRATEGY_ORDER = ["pre", "post"]

STRATEGY_LABELS = {"pre": "pre-filter", "post": "post-filter"}

STRATEGY_MARKERS = {"pre": "o", "post": "s"}

STRATEGY_COLOURS = {
    "light": {"pre": "#2a78d6", "post": "#eb6834"},
    "dark": {"pre": "#3987e5", "post": "#d95926"},
}

Row = Mapping[str, object]


def unique_value(rows: Sequence[Row], field: str) -> str:
    values = {str(row[field]) for row in rows}
    if len(values) != 1:
        raise SystemExit(f"rows mix {field} values {sorted(values)}; plot one configuration")
    return values.pop()


def series(rows: Sequence[Row], strategy: str, field: str) -> tuple[list[float], list[float]]:
    points = sorted(
        (float(row["selectivity"]), float(row[field]))
        for row in rows
        if row["strategy"] == strategy
    )
    return [x for x, _ in points], [y for _, y in points]


def style_axes(axes: Axes, theme: Mapping[str, object], selectivities: list[float]) -> None:
    axes.set_facecolor(theme["surface"])
    axes.set_xscale("log")
    axes.set_xticks(selectivities)
    axes.set_xticklabels([f"{value:.0%}" for value in selectivities])
    axes.minorticks_off()
    axes.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(which="both", colors=theme["muted"], labelsize=10, length=4)
    axes.set_xlabel(
        "filter selectivity (share of documents allowed)",
        color=theme["secondary"],
        fontsize=10.5,
        labelpad=8,
    )


def draw_strategies(
    axes: Axes, rows: Sequence[Row], field: str, theme_name: str, label_ends: bool
) -> None:
    theme = THEMES[theme_name]
    for strategy in STRATEGY_ORDER:
        xs, ys = series(rows, strategy, field)
        if not xs:
            continue
        colour = STRATEGY_COLOURS[theme_name][strategy]
        axes.plot(
            xs,
            ys,
            marker=STRATEGY_MARKERS[strategy],
            markersize=8,
            linewidth=2,
            color=colour,
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
            label=STRATEGY_LABELS[strategy],
            zorder=3,
        )
        if label_ends:
            axes.annotate(
                STRATEGY_LABELS[strategy],
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(9, 0),
                va="center",
                fontsize=9.5,
                color=theme["secondary"],
                zorder=4,
            )


def render(rows: Sequence[Row], theme_name: str, output_path: str) -> None:
    theme = THEMES[theme_name]
    selectivities = sorted({float(row["selectivity"]) for row in rows})
    k = unique_value(rows, "k")

    figure, (recall_axes, latency_axes) = plt.subplots(1, 2, figsize=(10.0, 4.6), dpi=160)
    figure.patch.set_facecolor(theme["surface"])

    for axes in (recall_axes, latency_axes):
        style_axes(axes, theme, selectivities)

    draw_strategies(recall_axes, rows, "recall_at_k", theme_name, label_ends=False)
    recall_axes.set_ylim(0.0, 1.05)
    recall_axes.set_ylabel(f"recall@{k}", color=theme["secondary"], fontsize=10.5, labelpad=8)
    recall_axes.set_title(
        f"Recall@{k}", color=theme["primary"], fontsize=12, loc="left", pad=10
    )

    draw_strategies(latency_axes, rows, "strategy_p50_ms", theme_name, label_ends=True)
    latency_axes.set_yscale("log")
    plain_number = FuncFormatter(lambda value, _: f"{value:g}")
    latency_axes.yaxis.set_major_formatter(plain_number)
    latency_axes.yaxis.set_minor_formatter(NullFormatter())
    latency_axes.set_ylabel(
        "median filter + search time (ms)", color=theme["secondary"], fontsize=10.5, labelpad=8
    )
    latency_axes.set_title(
        "Latency, excluding the log commit",
        color=theme["primary"],
        fontsize=12,
        loc="left",
        pad=10,
    )
    latency_axes.set_xlim(right=max(selectivities) * 2.2)

    legend = recall_axes.legend(
        loc="lower right", frameon=False, fontsize=10, labelcolor=theme["secondary"]
    )
    legend.set_zorder(5)

    subtitle = (
        f"{unique_value(rows, 'dataset')} · {int(unique_value(rows, 'num_base')):,} vectors · "
        f"{int(unique_value(rows, 'num_queries')):,} queries · k={k} · "
        f"ef_search={unique_value(rows, 'ef_search')} · "
        f"post widening ×{unique_value(rows, 'post_widening')} · "
        f"{unique_value(rows, 'threads')} thread · median of {unique_value(rows, 'runs')} runs"
    )
    figure.suptitle(
        "Filtered search: pre-filter vs post-filter",
        x=0.012,
        y=1.0,
        ha="left",
        va="top",
        color=theme["primary"],
        fontsize=15,
    )
    figure.text(0.012, 0.925, subtitle, va="top", ha="left", color=theme["muted"], fontsize=9.5)

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(figure)


def plot_filter_results(rows: Sequence[Row], out_prefix: str) -> list[str]:
    directory = os.path.dirname(out_prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)

    paths = [f"{out_prefix}.png", f"{out_prefix}_dark.png"]
    render(rows, "light", paths[0])
    render(rows, "dark", paths[1])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out", default="bench/results/filter_selectivity")
    arguments = parser.parse_args()

    with open(arguments.csv, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {arguments.csv}")

    for path in plot_filter_results(rows, arguments.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
