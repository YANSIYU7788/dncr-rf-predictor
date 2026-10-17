"""Probability-scale force plot for the exact five-feature Shapley decomposition."""
from __future__ import annotations

from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from rf_inference import FEATURE_LABELS

RED = "#FF0D57"
BLUE = "#008BFB"
DARK = "#262730"


def force_plot(result: dict) -> Figure:
    values = np.asarray(result["shap_values"], dtype=float)
    predicted = result["probability"]
    base = result["base_value"]
    positive = sorted(np.where(values > 1e-14)[0], key=lambda i: -values[i])
    negative = sorted(np.where(values < -1e-14)[0], key=lambda i: values[i])
    left = min(predicted - values[values > 0].sum(), base, predicted)
    right = max(predicted - values[values < 0].sum(), base, predicted)
    span = max(right - left, 0.12)
    xmin, xmax = left - span * 0.16, right + span * 0.16
    figure = Figure(figsize=(11.2, 4.0), facecolor="white", layout=None)
    axis = figure.add_axes([0.055, 0.48, 0.89, 0.34])
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(-0.08, 1.35)
    axis.set_yticks([])
    for side in ("left", "right", "top"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0.90))
    axis.spines["bottom"].set_color("#CAD0D8")
    axis.tick_params(axis="x", colors="#6B7280", labelsize=9, length=3, pad=5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for indices, sign, color in ((positive, 1, RED), (negative, -1, BLUE)):
        cursor = predicted
        for i in indices:
            width = abs(values[i])
            tip = min(width * 0.22, span * 0.018)
            end = cursor - sign * width
            if sign > 0:
                vertices = [(end, .22), (cursor-tip, .22), (cursor, .41), (cursor-tip, .60), (end, .60), (end+tip, .41)]
            else:
                vertices = [(end, .22), (cursor+tip, .22), (cursor, .41), (cursor+tip, .60), (end, .60), (end-tip, .41)]
            axis.add_patch(Polygon(vertices, closed=True, facecolor=color, edgecolor="white", linewidth=1))
            if width > span * .06:
                axis.text((end+cursor)/2, .41, str(i+1), color="white", fontsize=10, fontweight="bold", ha="center", va="center")
            cursor = end
    axis.plot([predicted, predicted], [.14, 1.04], color=DARK, linewidth=1.1)
    axis.plot([base, base], [.64, 1.04], color="#7D8797", linewidth=1, linestyle="--")
    separation = abs(predicted-base) / (xmax-xmin)
    if separation < .17:
        base_align = "right" if base <= predicted else "left"
        base_dx = -9 if base <= predicted else 9
        pred_align = "left" if base <= predicted else "right"
        pred_dx = 9 if base <= predicted else -9
    else:
        base_align = pred_align = "center"
        base_dx = pred_dx = 0
    axis.annotate(f"base value\n{base:.4f}", (base, 1.05), xytext=(base_dx, 3), textcoords="offset points", color="#737D8B", ha=base_align, va="bottom", fontsize=10)
    axis.annotate(f"f(x)\n{predicted:.4f}", (predicted, 1.05), xytext=(pred_dx, 3), textcoords="offset points", color=DARK, ha=pred_align, va="bottom", fontsize=11, fontweight="bold")
    figure.text(.495, .94, "higher  →", color=RED, ha="right", fontsize=11)
    figure.text(.505, .94, "←  lower", color=BLUE, ha="left", fontsize=11)
    figure.text(.5, .435, "Model output (probability scale)", color="#687386", ha="center", fontsize=9)
    for i, feature in enumerate(result["features"]):
        x = .055 + i * .186
        color = RED if values[i] > 1e-14 else BLUE if values[i] < -1e-14 else "#737D8B"
        raw = result["inputs"][i]
        used = result["model_inputs"][i]
        label = "Yes" if feature == "Depression" and raw == 1 else "No" if feature == "Depression" else f"{raw:g}"
        figure.text(x, .28, f"{i+1}. {FEATURE_LABELS[feature]}", fontsize=10, color=color, fontweight="bold")
        figure.text(x, .215, f"Value: {label}", fontsize=9.5, color=DARK)
        figure.text(x, .150, f"SHAP: {values[i]:+.4f}", fontsize=9.5, color=color)
        if abs(raw-used) > 1e-10:
            figure.text(x, .085, f"Model input: {used:.2f}", fontsize=8, color="#737D8B")
    return figure


def figure_bytes(figure: Figure, kind: str = "png") -> bytes:
    if kind not in ("png", "pdf", "tiff"):
        raise ValueError("Unsupported image format.")
    stream = BytesIO()
    kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if kind == "tiff" else {}
    figure.savefig(stream, format=kind, dpi=300, facecolor="white", **kwargs)
    return stream.getvalue()
