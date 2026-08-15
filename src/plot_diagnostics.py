from __future__ import annotations

import argparse
from pathlib import Path

from diagnostics import generate_diagnostics
from hamiltonian import FeynmanClockModel
from io_utils import create_run_dir, load_yaml, save_yaml
from simulation import BallisticSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static diagnostics for the Feynman ballistic-computer demo")
    parser.add_argument("--config", default="configs/rnot_5pcs.yaml")
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    p = cfg["physics"]
    d = cfg["diagnostics"]
    model = FeynmanClockModel(
        cursor_sites=int(p["cursor_sites"]),
        gate_names=tuple(p["gates"]),
        coupling=float(p["coupling"]),
        hbar=float(p["hbar"]),
        answer_initial=int(p["answer_initial"]),
    )
    sim = BallisticSimulator(model)
    run_dir = create_run_dir(args.output_dir)
    save_yaml(run_dir / "config_resolved.yaml", cfg)
    generate_diagnostics(
        sim,
        run_dir,
        float(d["t_start"]),
        float(d["t_end"]),
        int(d["num_points"]),
    )
    print(f"Diagnostics written to: {Path(run_dir).resolve()}")


if __name__ == "__main__":
    main()
