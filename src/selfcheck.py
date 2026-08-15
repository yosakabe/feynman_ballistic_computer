from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from hamiltonian import FeynmanClockModel, X, sqrt_not
from simulation import BallisticSimulator


def main() -> None:
    rnot = sqrt_not()
    assert np.allclose(rnot @ rnot, X, atol=1e-12), "sqrt(NOT)^2 != NOT"

    model = FeynmanClockModel(
        cursor_sites=5,
        gate_names=("sqrt_not", "sqrt_not", "identity", "identity"),
        coupling=1.0,
        hbar=1.0,
        answer_initial=0,
    )
    H = model.hamiltonian()
    assert H.shape == (64, 64)
    assert np.allclose(H, H.conj().T, atol=1e-12), "Hamiltonian is not Hermitian"

    psi0 = model.initial_state()
    assert np.isclose(np.linalg.norm(psi0), 1.0)

    sim = BallisticSimulator(model)
    snap0 = sim.snapshot(0.0)
    assert np.allclose(snap0.pcs_probabilities, [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(snap0.answer_bloch, [0.0, 0.0, 1.0], atol=1e-12)

    # The two √NOT gates complete NOT at PCS 2. Identity padding preserves it
    # through PCS 3 and PCS 4.
    for site in (2, 3, 4):
        ket = model.expected_answer_state_at_site(site)
        assert np.allclose(np.abs(ket), [0.0, 1.0], atol=1e-12), f"PCS {site} should encode |1>"

    # Spectral implementation agrees with a direct matrix exponential.
    t_test = 0.731
    psi_spectral = sim.state_at(t_test)
    psi_expm = expm(-1j * H * t_test / model.hbar) @ psi0
    assert np.allclose(psi_spectral, psi_expm, atol=1e-10)

    # Unitary evolution preserves normalization and one-cursor probability.
    for t in np.linspace(0, 12, 49):
        psi = sim.state_at(float(t))
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-10)
        assert np.isclose(sim.pcs_probabilities(psi).sum(), 1.0, atol=1e-10)

    # A measurement at t=0 must return PCS 0 and re-anchor without changing state.
    rng = np.random.default_rng(123)
    site, collapsed = sim.measure_pcs(0.0, rng)
    assert site == 0
    assert np.allclose(collapsed, psi0, atol=1e-12)

    print("Self-check passed (v3).")
    print(f"Hilbert dimension: {model.hilbert_dim}")
    print(f"Hermiticity error: {np.linalg.norm(H - H.conj().T):.3e}")
    print(f"sqrt(NOT)^2 error: {np.linalg.norm(rnot @ rnot - X):.3e}")


if __name__ == "__main__":
    main()
