from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from simulation import SimulationSnapshot


def make_pcs_figure(
    snapshot: SimulationSnapshot,
    gate_labels: list[str],
    measured_site: int | None = None,
) -> go.Figure:
    """Create the initial PCS figure.

    Trace indices are intentionally stable because app.py uses Dash Patch to
    update only the dynamic fields on subsequent frames:
      0: probability bars
      1: PCS chain / probability-sized markers
      2: last measured cursor (star)
    """
    n = len(snapshot.pcs_probabilities)
    x = np.arange(n, dtype=float)
    p = snapshot.pcs_probabilities

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=p,
            width=0.48,
            name="PCS probability",
            text=[f"{100*v:.1f}%" for v in p],
            textposition="outside",
            hovertemplate="PCS %{x}<br>P=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.full(n, 1.24),
            mode="lines+markers+text",
            marker={"size": (18 + 28 * p).tolist()},
            line={"width": 3},
            text=[f"PCS {i}" for i in range(n)],
            textposition="top center",
            name="quantum cursor",
            hovertemplate="PCS %{x}<extra></extra>",
        )
    )
    star_x = [] if measured_site is None else [measured_site]
    star_y = [] if measured_site is None else [1.10]
    fig.add_trace(
        go.Scatter(
            x=star_x,
            y=star_y,
            mode="markers",
            marker={"size": 25, "symbol": "star"},
            name="last measured cursor",
            hovertemplate="Last measured: PCS %{x}<extra></extra>",
        )
    )

    for i, label in enumerate(gate_labels):
        fig.add_annotation(
            x=i + 0.5,
            y=1.38,
            text=label,
            showarrow=False,
            font={"size": 15},
        )

    fig.update_layout(
        title=f"Quantum cursor / PCS distribution    t = {snapshot.t:.3f}",
        xaxis={"tickmode": "array", "tickvals": list(range(n)), "range": [-0.55, n - 0.45]},
        yaxis={"title": "Probability", "range": [0.0, 1.58]},
        showlegend=False,
        margin={"l": 55, "r": 25, "t": 75, "b": 45},
        height=390,
        uirevision="pcs-v3",
    )
    return fig


def make_bloch_figure(snapshot: SimulationSnapshot) -> go.Figure:
    """Create the initial Bloch-sphere figure.

    The sphere is static.  app.py subsequently patches only trace 1 (the
    Bloch vector) and the title, avoiding reconstruction/transfer of the 3-D
    surface on every animation tick.
    """
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 28)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    r = snapshot.answer_bloch
    purity = float(np.real(np.trace(snapshot.answer_rho @ snapshot.answer_rho)))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.16,
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, float(r[0])],
            y=[0.0, float(r[1])],
            z=[0.0, float(r[2])],
            mode="lines+markers",
            line={"width": 7},
            marker={"size": [2, 6]},
            name="Answer Bloch vector",
            hovertemplate="(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[1.08, -1.08],
            mode="text",
            text=["|0⟩", "|1⟩"],
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Answer bit — reduced state (purity={purity:.3f})",
        scene={
            "xaxis": {"range": [-1.15, 1.15], "title": "X"},
            "yaxis": {"range": [-1.15, 1.15], "title": "Y"},
            "zaxis": {"range": [-1.15, 1.15], "title": "Z"},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        height=390,
        showlegend=False,
        uirevision="bloch-v3",
    )
    return fig


def conditional_answer_rows(snapshot: SimulationSnapshot) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for site, (p_cursor, p1, bloch) in enumerate(
        zip(
            snapshot.pcs_probabilities,
            snapshot.conditional_answer_prob1,
            snapshot.conditional_bloch,
        )
    ):
        if np.isnan(p1):
            rows.append(
                {
                    "PCS": str(site),
                    "P(PCS)": f"{p_cursor:.4f}",
                    "P(Answer=1 | PCS)": "—",
                    "conditional Bloch": "—",
                }
            )
        else:
            rows.append(
                {
                    "PCS": str(site),
                    "P(PCS)": f"{p_cursor:.4f}",
                    "P(Answer=1 | PCS)": f"{p1:.4f}",
                    "conditional Bloch": f"({bloch[0]:+.3f}, {bloch[1]:+.3f}, {bloch[2]:+.3f})",
                }
            )
    return rows
