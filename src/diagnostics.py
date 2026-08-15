from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from simulation import BallisticSimulator


def generate_diagnostics(
    simulator: BallisticSimulator,
    run_dir: str | Path,
    t_start: float,
    t_end: float,
    num_points: int,
) -> None:
    run_dir = Path(run_dir)
    times = np.linspace(t_start, t_end, num_points)
    pcs_series = []
    bloch_series = []
    conditional_p1_series = []

    for t in times:
        snap = simulator.snapshot(float(t))
        pcs_series.append(snap.pcs_probabilities)
        bloch_series.append(snap.answer_bloch)
        conditional_p1_series.append(snap.conditional_answer_prob1)

    pcs = np.asarray(pcs_series)
    bloch = np.asarray(bloch_series)
    conditional_p1 = np.asarray(conditional_p1_series)

    with (run_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = (
            ["t"]
            + [f"pcs_{i}" for i in range(pcs.shape[1])]
            + ["bloch_x", "bloch_y", "bloch_z"]
            + [f"conditional_answer_p1_pcs_{i}" for i in range(pcs.shape[1])]
        )
        writer.writerow(header)
        for i, t in enumerate(times):
            writer.writerow(
                [float(t), *pcs[i].tolist(), *bloch[i].tolist(), *conditional_p1[i].tolist()]
            )

    fig_pcs = go.Figure()
    for site in range(pcs.shape[1]):
        fig_pcs.add_trace(go.Scatter(x=times, y=pcs[:, site], mode="lines", name=f"PCS {site}"))
    fig_pcs.update_layout(
        title="PCS probability evolution (unitary, no measurement)",
        xaxis_title="t",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
    )
    fig_pcs.write_html(run_dir / "pcs_probability_evolution.html", include_plotlyjs="cdn")

    fig_bloch = go.Figure()
    for axis, label in enumerate(["X", "Y", "Z"]):
        fig_bloch.add_trace(go.Scatter(x=times, y=bloch[:, axis], mode="lines", name=label))
    fig_bloch.update_layout(
        title="Reduced Answer-bit Bloch components",
        xaxis_title="t",
        yaxis_title="Bloch component",
        yaxis_range=[-1.05, 1.05],
    )
    fig_bloch.write_html(run_dir / "answer_bloch_evolution.html", include_plotlyjs="cdn")

    fig_cond = go.Figure()
    for site in range(conditional_p1.shape[1]):
        fig_cond.add_trace(
            go.Scatter(
                x=times,
                y=conditional_p1[:, site],
                mode="lines",
                name=f"PCS {site}",
            )
        )
    fig_cond.update_layout(
        title="Conditional Answer probability P(Answer=1 | PCS=i)",
        xaxis_title="t",
        yaxis_title="Conditional probability",
        yaxis_range=[0, 1],
    )
    fig_cond.write_html(
        run_dir / "conditional_answer_evolution.html", include_plotlyjs="cdn"
    )

