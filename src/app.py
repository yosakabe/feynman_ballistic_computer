from __future__ import annotations

import argparse
import time
from pathlib import Path

from dash import Dash, Input, Output, Patch, State, ctx, dcc, html
import numpy as np

from diagnostics import generate_diagnostics
from hamiltonian import FeynmanClockModel, X, sqrt_not
from io_utils import EventLogger, create_run_dir, load_yaml, save_yaml
from simulation import BallisticSimulator
from visualization import conditional_answer_rows, make_bloch_figure, make_pcs_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Feynman ballistic-computer demo v3")
    parser.add_argument("--config", default="configs/rnot_5pcs.yaml", help="YAML config path")
    parser.add_argument("--output-dir", default="output", help="Root directory for timestamped output")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    return parser.parse_args()


def build_model(cfg: dict) -> FeynmanClockModel:
    p = cfg["physics"]
    return FeynmanClockModel(
        cursor_sites=int(p["cursor_sites"]),
        gate_names=tuple(p["gates"]),
        coupling=float(p["coupling"]),
        hbar=float(p["hbar"]),
        answer_initial=int(p["answer_initial"]),
    )


def write_run_metadata(run_dir: Path, cfg: dict, model: FeynmanClockModel, simulator: BallisticSimulator) -> None:
    save_yaml(run_dir / "config_resolved.yaml", cfg)
    rnot = sqrt_not()
    derived = {
        "version": 3,
        "num_qubits": model.num_qubits,
        "cursor_sites": model.cursor_sites,
        "answer_qubits": 1,
        "hilbert_dimension": model.hilbert_dim,
        "gate_names": list(model.gate_names),
        "completion_site": int(cfg["physics"].get("completion_site", model.cursor_sites - 1)),
        "hamiltonian_shape": list(simulator.H.shape),
        "hamiltonian_hermiticity_error_fro": float(np.linalg.norm(simulator.H - simulator.H.conj().T)),
        "sqrt_not_squared_error_fro": float(np.linalg.norm(rnot @ rnot - X)),
        "eigenvalue_min": float(simulator.eigenvalues.min()),
        "eigenvalue_max": float(simulator.eigenvalues.max()),
        "final_answer_state_prob1": float(abs(model.expected_answer_state_at_site(model.cursor_sites - 1)[1]) ** 2),
    }
    save_yaml(run_dir / "derived_parameters.yaml", derived)


def gate_display_name(name: str) -> str:
    key = name.lower()
    if key in {"sqrt_not", "sqrt-not", "rnot", "sqrtx"}:
        return "√NOT"
    if key in {"not", "x"}:
        return "NOT"
    if key in {"identity", "i", "id"}:
        return "I (padding)"
    return name


def _history_text(history: list[dict]) -> str:
    if not history:
        return "Measured cursor history: —"
    path = " → ".join(str(item["site"]) for item in history)
    return f"Measured cursor history: {path}"


def make_app(cfg: dict, simulator: BallisticSimulator, logger: EventLogger, rng: np.random.Generator) -> Dash:
    app_cfg = cfg["app"]
    ui_cfg = cfg["ui"]
    gate_labels = [gate_display_name(g) for g in simulator.model.gate_names]
    initial_mode = str(ui_cfg.get("initial_evolution_mode", "unobserved"))
    initial_measurement_interval = float(ui_cfg.get("initial_measurement_interval", 0.75))
    history_limit = int(ui_cfg.get("measurement_history_length", 24))
    completion_site = int(cfg["physics"].get("completion_site", simulator.model.cursor_sites - 1))
    max_measurements_per_tick = int(ui_cfg.get("max_measurements_per_tick", 20))

    app = Dash(__name__)
    app.title = str(app_cfg["title"])

    initial_clock = {
        "t": 0.0,
        "running": False,
        "last_wall": time.time(),
        "last_measurement": None,
        "measurement_history": [],
        "mode": initial_mode,
        "next_measurement_t": initial_measurement_interval if initial_mode == "periodic" else None,
    }

    initial_snapshot = simulator.snapshot(0.0)
    initial_pcs_fig = make_pcs_figure(initial_snapshot, gate_labels)
    initial_bloch_fig = make_bloch_figure(initial_snapshot)

    app.layout = html.Div(
        [
            html.H1(app_cfg["title"], style={"marginBottom": "0.25rem"}),
            dcc.Store(id="clock-store", data=initial_clock),
            # v3 retains the v2 fix: the timer is genuinely stopped while paused.  This prevents
            # animation callbacks from continuously competing with button clicks.
            dcc.Interval(
                id="tick",
                interval=int(ui_cfg["interval_ms"]),
                n_intervals=0,
                disabled=True,
            ),
            html.Div(
                [
                    html.Button("▶ Play", id="play-btn", n_clicks=0),
                    html.Button("Pause", id="pause-btn", n_clicks=0, style={"marginLeft": "0.5rem"}),
                    html.Button("Reset", id="reset-btn", n_clicks=0, style={"marginLeft": "0.5rem"}),
                    html.Button("Measure PCS now", id="measure-btn", n_clicks=0, style={"marginLeft": "0.5rem"}),
                ],
                style={"marginBottom": "0.85rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Evolution mode", style={"fontWeight": "600", "marginBottom": "0.25rem"}),
                            dcc.RadioItems(
                                id="mode-radio",
                                options=[
                                    {"label": " Unobserved unitary evolution", "value": "unobserved"},
                                    {"label": " Periodic PCS measurement", "value": "periodic"},
                                ],
                                value=initial_mode,
                                inline=False,
                            ),
                        ],
                        style={"flex": "1 1 330px"},
                    ),
                    html.Div(
                        [
                            html.Div("Measurement interval (simulation time)", style={"fontWeight": "600"}),
                            dcc.Slider(
                                id="measurement-interval-slider",
                                min=float(ui_cfg.get("min_measurement_interval", 0.15)),
                                max=float(ui_cfg.get("max_measurement_interval", 3.0)),
                                step=0.05,
                                value=initial_measurement_interval,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"flex": "1 1 330px"},
                    ),
                    html.Div(
                        [
                            html.Div("Simulation speed", style={"fontWeight": "600"}),
                            dcc.Slider(
                                id="speed-slider",
                                min=0.1,
                                max=float(ui_cfg["max_speed"]),
                                step=0.1,
                                value=float(ui_cfg["initial_speed"]),
                                marks={0.5: "0.5×", 1.0: "1×", 2.0: "2×", float(ui_cfg["max_speed"]): f"{ui_cfg['max_speed']}×"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={"flex": "1 1 330px"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "0.8rem"},
            ),
            html.Div(id="status-text", style={"fontWeight": "600", "marginBottom": "0.3rem"}),
            html.Div(id="measurement-history", style={"fontFamily": "monospace", "marginBottom": "0.5rem"}),
            dcc.Graph(
                id="pcs-graph",
                figure=initial_pcs_fig,
                config={"displayModeBar": False},
                style={"width": "100%"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="bloch-graph",
                                figure=initial_bloch_fig,
                                config={"displayModeBar": False},
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1 1 42%", "minWidth": "380px"},
                    ),
                    html.Div(
                        [
                            html.H3(
                                "Answer bit conditioned on the observed PCS site",
                                style={"marginTop": "0.35rem", "marginBottom": "0.75rem"},
                            ),
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("PCS"),
                                                html.Th("P(PCS)"),
                                                html.Th("P(Answer=1 | PCS)"),
                                                html.Th("conditional Bloch"),
                                            ]
                                        )
                                    ),
                                    html.Tbody(id="conditional-table-body"),
                                ],
                                style={
                                    "width": "100%",
                                    "borderCollapse": "collapse",
                                    "textAlign": "center",
                                    "fontFamily": "monospace",
                                    "fontSize": "0.9rem",
                                },
                            ),
                        ],
                        style={"flex": "1 1 54%", "minWidth": "520px", "paddingTop": "0.25rem"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "1rem",
                    "alignItems": "flex-start",
                    "flexWrap": "wrap",
                    "marginTop": "0.35rem",
                },
            ),
            html.Details(
                [
                    html.Summary("Model Note"),
                    dcc.Markdown(
                        f"""
PCS (Program Counter Site) を {simulator.model.cursor_sites} 個に拡張した Feynman clock です。最初の2遷移で Answer bit に √NOT を2回作用させ、その後の2サイトは計算結果を保持する padding site としています。

The model uses **{simulator.model.cursor_sites} PCS qubits + 1 Answer qubit = {simulator.model.num_qubits} qubits**, so the full state vector has dimension {simulator.model.hilbert_dim}.  
The initial state has only PCS 0 occupied. The Hamiltonian is

$$
H = J\\sum_i\\left(q_{{i+1}}^\\dagger q_i A_{{i+1}} + q_i^\\dagger q_{{i+1}} A_{{i+1}}^\\dagger\\right).
$$

In v2, $A_1=A_2=\\sqrt{{X}}$ and $A_3=A_4=I$. Thus PCS 2--4 all correspond to a completed NOT operation on the Answer bit. The identity gates are padding clock steps: they change cursor dynamics without changing the completed answer.
                        """,
                        mathjax=True,
                    ),
                ],
                style={"marginTop": "1rem"},
            ),
        ],
        style={"maxWidth": "1180px", "margin": "0 auto", "padding": "1rem 1.25rem", "fontFamily": "Arial, sans-serif"},
    )

    @app.callback(
        Output("clock-store", "data"),
        Output("tick", "disabled"),
        Input("tick", "n_intervals"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("measure-btn", "n_clicks"),
        Input("mode-radio", "value"),
        Input("measurement-interval-slider", "value"),
        State("clock-store", "data"),
        State("speed-slider", "value"),
        prevent_initial_call=False,
    )
    def update_clock(_tick, _play, _pause, _reset, _measure, mode, measurement_interval, clock, speed):
        clock = dict(clock or initial_clock)
        now = time.time()
        running = bool(clock.get("running", False))
        last_wall = float(clock.get("last_wall", now))
        t = float(clock.get("t", 0.0))
        speed = float(speed or 1.0)
        measurement_interval = max(1e-6, float(measurement_interval or initial_measurement_interval))
        history = list(clock.get("measurement_history", []))
        message = clock.get("last_measurement")
        next_measurement_t = clock.get("next_measurement_t")
        previous_mode = str(clock.get("mode", initial_mode))
        mode = str(mode or previous_mode)

        # Account for elapsed wall time first.  State is reconstructed from the
        # simulation time, never by accumulating a numerical dt at render rate.
        if running:
            t += max(0.0, now - last_wall) * speed

        trigger = ctx.triggered_id

        def record_measurement(site: int, measure_t: float, source: str) -> None:
            nonlocal message, history
            done = site >= completion_site
            message = {"site": site, "done": done, "sim_time": measure_t, "source": source}
            history.append({"site": site, "sim_time": measure_t, "source": source})
            history = history[-history_limit:]
            logger.log(
                "measure_pcs",
                sim_time=measure_t,
                site=site,
                source=source,
                calculation_completed=done,
            )

        if trigger == "play-btn":
            running = True
            if mode == "periodic" and next_measurement_t is None:
                next_measurement_t = t + measurement_interval
            logger.log("play", sim_time=t, speed=speed, mode=mode)

        elif trigger == "pause-btn":
            running = False
            logger.log("pause", sim_time=t)

        elif trigger == "reset-btn":
            simulator.reset()
            t = 0.0
            running = False
            message = None
            history = []
            next_measurement_t = measurement_interval if mode == "periodic" else None
            logger.log("reset", sim_time=t, mode=mode)

        elif trigger == "measure-btn":
            site, _ = simulator.measure_pcs(t, rng)
            record_measurement(site, t, "manual")
            if mode == "periodic":
                next_measurement_t = t + measurement_interval

        elif trigger == "mode-radio":
            if mode == "periodic":
                next_measurement_t = t + measurement_interval
            else:
                next_measurement_t = None
            logger.log("mode_change", sim_time=t, mode=mode)

        elif trigger == "measurement-interval-slider":
            if mode == "periodic":
                next_measurement_t = t + measurement_interval
            logger.log("measurement_interval_change", sim_time=t, interval=measurement_interval)

        elif trigger == "tick" and running and mode == "periodic":
            if next_measurement_t is None:
                next_measurement_t = t + measurement_interval

            count = 0
            while next_measurement_t <= t + 1e-12 and count < max_measurements_per_tick:
                measure_t = float(next_measurement_t)
                site, _ = simulator.measure_pcs(measure_t, rng)
                record_measurement(site, measure_t, "periodic")
                next_measurement_t += measurement_interval
                count += 1

            # If the UI was suspended for a long time, avoid a huge backlog.
            if count >= max_measurements_per_tick and next_measurement_t <= t:
                next_measurement_t = t + measurement_interval
                logger.log("measurement_backlog_skipped", sim_time=t)

        # If the mode was altered indirectly (e.g. restored browser state), keep
        # scheduling coherent even when ctx does not report mode-radio.
        if mode != previous_mode and trigger != "mode-radio":
            next_measurement_t = t + measurement_interval if mode == "periodic" else None

        disabled = not running
        return (
            {
                "t": t,
                "running": running,
                "last_wall": now,
                "last_measurement": message,
                "measurement_history": history,
                "mode": mode,
                "next_measurement_t": next_measurement_t,
            },
            disabled,
        )

    @app.callback(
        Output("pcs-graph", "figure"),
        Output("bloch-graph", "figure"),
        Output("conditional-table-body", "children"),
        Output("status-text", "children"),
        Output("measurement-history", "children"),
        Input("clock-store", "data"),
    )
    def render(clock):
        t = float(clock["t"])
        snap = simulator.snapshot(t)
        meas = clock.get("last_measurement")
        measured_site = None if meas is None else int(meas["site"])

        # v3 retains the partial-update performance fix: only patch the values that actually change.  The
        # 3-D Bloch sphere surface and static PCS annotations are not resent.
        pcs_patch = Patch()
        p = snap.pcs_probabilities
        pcs_patch["data"][0]["y"] = p.tolist()
        pcs_patch["data"][0]["text"] = [f"{100*v:.1f}%" for v in p]
        pcs_patch["data"][1]["marker"]["size"] = (18 + 28 * p).tolist()
        pcs_patch["data"][2]["x"] = [] if measured_site is None else [measured_site]
        pcs_patch["data"][2]["y"] = [] if measured_site is None else [1.10]
        pcs_patch["layout"]["title"]["text"] = f"Quantum cursor / PCS distribution    t = {t:.3f}"

        r = snap.answer_bloch
        purity = float(np.real(np.trace(snap.answer_rho @ snap.answer_rho)))
        bloch_patch = Patch()
        bloch_patch["data"][1]["x"] = [0.0, float(r[0])]
        bloch_patch["data"][1]["y"] = [0.0, float(r[1])]
        bloch_patch["data"][1]["z"] = [0.0, float(r[2])]
        bloch_patch["layout"]["title"]["text"] = f"Answer bit — reduced state (purity={purity:.3f})"

        table_data = conditional_answer_rows(snap)
        table = [
            html.Tr(
                [
                    html.Td(row["PCS"], style={"padding": "0.45rem", "borderTop": "1px solid #ddd"}),
                    html.Td(row["P(PCS)"], style={"padding": "0.45rem", "borderTop": "1px solid #ddd"}),
                    html.Td(row["P(Answer=1 | PCS)"], style={"padding": "0.45rem", "borderTop": "1px solid #ddd"}),
                    html.Td(row["conditional Bloch"], style={"padding": "0.45rem", "borderTop": "1px solid #ddd"}),
                ]
            )
            for row in table_data
        ]

        mode_label = "UNOBSERVED" if clock.get("mode") == "unobserved" else "PERIODIC MEASUREMENT"
        status = f"t = {t:.3f}    |    {'RUNNING' if clock['running'] else 'PAUSED'}    |    {mode_label}"
        if meas is not None:
            source = meas.get("source", "manual")
            if meas["done"]:
                status += f"    |    last measurement ({source}): PCS {meas['site']} → completed-answer region"
            else:
                status += f"    |    last measurement ({source}): PCS {meas['site']}"

        return pcs_patch, bloch_patch, table, status, _history_text(clock.get("measurement_history", []))

    return app


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model = build_model(cfg)
    simulator = BallisticSimulator(model)

    run_dir = create_run_dir(args.output_dir)
    write_run_metadata(run_dir, cfg, model, simulator)
    logger = EventLogger(run_dir / "events.jsonl")
    logger.log("start", version=3, config=str(Path(args.config).resolve()), run_dir=str(run_dir.resolve()))

    diagnostics_cfg = cfg.get("diagnostics", {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True)) and not args.no_diagnostics
    if diagnostics_enabled:
        generate_diagnostics(
            simulator,
            run_dir,
            float(diagnostics_cfg.get("t_start", 0.0)),
            float(diagnostics_cfg.get("t_end", 18.0)),
            int(diagnostics_cfg.get("num_points", 361)),
        )

    seed = int(cfg.get("random", {}).get("seed", 0))
    rng = np.random.default_rng(seed)
    app = make_app(cfg, simulator, logger, rng)

    host = args.host or cfg["app"].get("host", "127.0.0.1")
    port = args.port or int(cfg["app"].get("port", 8050))
    debug = bool(args.debug or cfg["app"].get("debug", False))

    print(f"Run directory: {run_dir.resolve()}")
    print(f"Open: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
