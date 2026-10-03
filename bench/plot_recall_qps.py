from __future__ import annotations

import argparse
import csv
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SERIES_ORDER = [
    "khoj-hnsw",
    "faiss-hnsw",
    "khoj-flat",
    "khoj-hnsw-baseline",
    "khoj-hnsw-native",
]

SERIES_LABELS = {
    "khoj-hnsw": "khoj HNSW",
    "faiss-hnsw": "FAISS HNSW",
    "khoj-flat": "khoj flat (exact)",
    "khoj-hnsw-baseline": "khoj HNSW, before optimization",
    "khoj-hnsw-native": "khoj HNSW, -march=native (non-default build)",
}

SERIES_MARKERS = {
    "khoj-hnsw": "o",
    "faiss-hnsw": "s",
    "khoj-flat": "D",
    "khoj-hnsw-baseline": "v",
    "khoj-hnsw-native": "^",
}

SERIES_LINESTYLES = {"khoj-hnsw-baseline": ":", "khoj-hnsw-native": "--"}

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {
            "khoj-hnsw": "#2a78d6",
            "faiss-hnsw": "#eb6834",
            "khoj-flat": "#1baf7a",
            "khoj-hnsw-baseline": "#eda100",
            "khoj-hnsw-native": "#e87ba4",
        },
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {
            "khoj-hnsw": "#3987e5",
            "faiss-hnsw": "#d95926",
            "khoj-flat": "#199e70",
            "khoj-hnsw-baseline": "#c98500",
            "khoj-hnsw-native": "#d55181",
        },
    },
}


def load_rows(arguments: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for argument in arguments:
        series, separator, path = argument.partition("=")
        if not separator:
            series, path = "", argument
        with open(path, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["series"] = series or f"{row['engine']}-{row['index']}"
                rows.append(row)
    if not rows:
        raise SystemExit("no rows found in the supplied CSV files")
    return rows


def group_series(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["series"]
        if key not in SERIES_ORDER:
            raise SystemExit(
                f"unmapped series {key}; add a validated colour slot before plotting it"
            )
        grouped.setdefault(key, []).append(
            {
                "recall": float(row["recall_at_k"]),
                "qps": float(row["qps_median"]),
                "ef_search": row["ef_search"],
            }
        )
    for points in grouped.values():
        points.sort(key=lambda point: point["recall"])
    return grouped


def unique_field(rows: list[dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise SystemExit(
            f"rows mix {field} values {sorted(values)}; filter the CSV so one plot "
            f"describes one {field}"
        )
    return values.pop()


def build_subtitle(rows: list[dict[str, str]]) -> str:
    dataset = unique_field(rows, "dataset")
    threads = unique_field(rows, "threads")
    runs = unique_field(rows, "runs")
    k = unique_field(rows, "k")
    num_base = unique_field(rows, "num_base")
    num_queries = unique_field(rows, "num_queries")
    return (
        f"{dataset} · {int(num_base):,} base × {int(num_queries):,} queries · "
        f"k={k} · {threads} thread(s) · median of {runs} runs"
    )


def render(
    grouped: dict[str, list[dict[str, Any]]],
    subtitle: str,
    k: str,
    theme_name: str,
    output_path: str,
) -> None:
    theme = THEMES[theme_name]

    figure, axes = plt.subplots(figsize=(9.0, 5.6), dpi=160)
    figure.patch.set_facecolor(theme["surface"])
    axes.set_facecolor(theme["surface"])

    axes.set_yscale("log")
    axes.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    axes.grid(True, which="minor", color=theme["grid"], linewidth=0.5, alpha=0.6, zorder=0)
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1.0)

    axes.tick_params(colors=theme["muted"], labelsize=10, length=4)

    for key in SERIES_ORDER:
        points = grouped.get(key)
        if not points:
            continue

        colour = theme["series"][key]
        recalls = [point["recall"] for point in points]
        rates = [point["qps"] for point in points]

        if len(points) == 1:
            axes.plot(
                recalls,
                rates,
                marker=SERIES_MARKERS[key],
                markersize=10,
                linestyle="none",
                color=colour,
                markeredgecolor=theme["surface"],
                markeredgewidth=2,
                label=SERIES_LABELS[key],
                zorder=3,
            )
            axes.annotate(
                "exact baseline",
                (recalls[0], rates[0]),
                textcoords="offset points",
                xytext=(-14, 0),
                ha="right",
                va="center",
                fontsize=9,
                color=theme["secondary"],
                zorder=4,
            )
        else:
            axes.plot(
                recalls,
                rates,
                marker=SERIES_MARKERS[key],
                markersize=8,
                linewidth=2,
                linestyle=SERIES_LINESTYLES.get(key, "-"),
                color=colour,
                markeredgecolor=theme["surface"],
                markeredgewidth=2,
                label=SERIES_LABELS[key],
                zorder=3,
            )

    all_recalls = [point["recall"] for points in grouped.values() for point in points]
    span = max(all_recalls) - min(all_recalls)
    padding = max(span * 0.06, 0.01)
    axes.set_xlim(min(all_recalls) - padding, max(all_recalls) + padding)
    axes.autoscale_view(scalex=False)

    # Every khoj HNSW variant shares one recall per ef_search, so a label above or
    # below a point lands on another curve's marker. Labels go beside the point
    # instead: right along gentle stretches, left where the curve falls steeply,
    # and right of the final point.
    labelled = grouped.get("khoj-hnsw", [])
    for index, point in enumerate(labelled):
        if index == len(labelled) - 1:
            offset, ha, va = (7, 0), "left", "center"
        else:
            here = axes.transData.transform((point["recall"], point["qps"]))
            following = labelled[index + 1]
            there = axes.transData.transform((following["recall"], following["qps"]))
            gentle = abs(there[1] - here[1]) < abs(there[0] - here[0])
            offset, ha, va = ((7, 4), "left", "center") if gentle else ((-7, 0), "right", "center")
        axes.annotate(
            point["ef_search"],
            (point["recall"], point["qps"]),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va=va,
            fontsize=8.5,
            color=theme["muted"],
            zorder=4,
        )

    axes.set_xlabel(f"recall@{k}", color=theme["secondary"], fontsize=11, labelpad=8)
    axes.set_ylabel("queries per second", color=theme["secondary"], fontsize=11, labelpad=8)

    axes.set_title(
        "Recall vs throughput",
        color=theme["primary"],
        fontsize=15,
        loc="left",
        pad=26,
    )
    axes.text(
        0.0,
        1.035,
        subtitle,
        transform=axes.transAxes,
        color=theme["muted"],
        fontsize=10,
        ha="left",
    )
    axes.text(
        0.0,
        -0.155,
        "labelled points are ef_search; upper right is better",
        transform=axes.transAxes,
        color=theme["muted"],
        fontsize=9,
        ha="left",
    )

    legend = axes.legend(
        loc="lower left",
        frameon=False,
        fontsize=10,
        labelcolor=theme["secondary"],
        handletextpad=0.6,
    )
    legend.set_zorder(5)

    figure.tight_layout()
    figure.savefig(output_path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="+",
        help="CSV path, optionally prefixed SERIES=path to override the "
        "engine-index series key (e.g. khoj-hnsw-native=results.csv)",
    )
    parser.add_argument("--out", default="bench/results/recall_vs_qps")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    rows = load_rows(arguments.csv)

    grouped = group_series(rows)
    subtitle = build_subtitle(rows)
    k = unique_field(rows, "k")

    directory = os.path.dirname(arguments.out)
    if directory:
        os.makedirs(directory, exist_ok=True)

    light_path = f"{arguments.out}.png"
    dark_path = f"{arguments.out}_dark.png"
    render(grouped, subtitle, k, "light", light_path)
    render(grouped, subtitle, k, "dark", dark_path)

    print(f"wrote {light_path}")
    print(f"wrote {dark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
