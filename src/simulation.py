from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

from hamiltonian import FeynmanClockModel, X, Y, Z


@dataclass
class SimulationSnapshot:
    t: float
    state: np.ndarray
    pcs_probabilities: np.ndarray
    answer_rho: np.ndarray
    answer_bloch: np.ndarray
    conditional_answer_prob1: np.ndarray
    conditional_bloch: np.ndarray


class BallisticSimulator:
    """Exact unitary evolution of the small Feynman-clock model.

    Time evolution is always computed from an anchor state and anchor time,
    rather than by repeatedly applying a small dt step.  This avoids numerical
    drift and makes Pause/Resume independent of rendering frequency.
    """

    def __init__(self, model: FeynmanClockModel):
        self.model = model
        self.H = model.hamiltonian()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
        self._lock = threading.RLock()
        self._initial_state = model.initial_state()
        self._anchor_state = self._initial_state.copy()
        self._anchor_time = 0.0

    def reset(self) -> None:
        with self._lock:
            self._anchor_state = self._initial_state.copy()
            self._anchor_time = 0.0

    def _evolve_from(self, psi0: np.ndarray, delta_t: float) -> np.ndarray:
        coeff = self.eigenvectors.conj().T @ psi0
        phase = np.exp(-1j * self.eigenvalues * delta_t / self.model.hbar)
        psi = self.eigenvectors @ (phase * coeff)
        norm = np.linalg.norm(psi)
        if norm == 0:
            raise RuntimeError("State vector norm became zero")
        return psi / norm

    def state_at(self, t: float) -> np.ndarray:
        with self._lock:
            if t < self._anchor_time - 1e-12:
                # Scrubbing into the past is defined relative to the original initial state.
                return self._evolve_from(self._initial_state, t)
            return self._evolve_from(self._anchor_state, t - self._anchor_time)

    def pcs_probabilities(self, psi: np.ndarray) -> np.ndarray:
        probs = []
        for site in range(self.model.cursor_sites):
            proj = self.model.cursor_projector(site)
            p = float(np.real(np.vdot(psi, proj @ psi)))
            probs.append(max(0.0, p))
        probs = np.array(probs, dtype=float)
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs

    def _answer_density_matrix(self, psi: np.ndarray) -> np.ndarray:
        # Answer is the final qubit: reshape [cursor basis, answer basis].
        cursor_dim = 2 ** self.model.cursor_sites
        psi_matrix = psi.reshape(cursor_dim, 2)
        rho = psi_matrix.conj().T @ psi_matrix
        return rho / np.trace(rho)

    @staticmethod
    def _bloch(rho: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.real(np.trace(rho @ X)),
                np.real(np.trace(rho @ Y)),
                np.real(np.trace(rho @ Z)),
            ],
            dtype=float,
        )

    def conditional_answer_density_matrix(self, psi: np.ndarray, site: int) -> tuple[float, np.ndarray | None]:
        proj = self.model.cursor_projector(site)
        projected = proj @ psi
        p = float(np.real(np.vdot(projected, projected)))
        if p < 1e-12:
            return p, None
        projected /= np.sqrt(p)
        return p, self._answer_density_matrix(projected)

    def snapshot(self, t: float) -> SimulationSnapshot:
        psi = self.state_at(t)
        pcs = self.pcs_probabilities(psi)
        answer_rho = self._answer_density_matrix(psi)
        answer_bloch = self._bloch(answer_rho)

        cond_prob1 = []
        cond_bloch = []
        for site in range(self.model.cursor_sites):
            _, rho = self.conditional_answer_density_matrix(psi, site)
            if rho is None:
                cond_prob1.append(np.nan)
                cond_bloch.append([np.nan, np.nan, np.nan])
            else:
                cond_prob1.append(float(np.real(rho[1, 1])))
                cond_bloch.append(self._bloch(rho))

        return SimulationSnapshot(
            t=t,
            state=psi,
            pcs_probabilities=pcs,
            answer_rho=answer_rho,
            answer_bloch=answer_bloch,
            conditional_answer_prob1=np.array(cond_prob1),
            conditional_bloch=np.array(cond_bloch),
        )

    def measure_pcs(self, t: float, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        """Projectively measure the PCS position and re-anchor the evolution."""
        with self._lock:
            psi = self.state_at(t)
            probs = self.pcs_probabilities(psi)
            site = int(rng.choice(np.arange(self.model.cursor_sites), p=probs))
            projected = self.model.cursor_projector(site) @ psi
            norm = np.linalg.norm(projected)
            if norm < 1e-14:
                raise RuntimeError("Measurement selected a zero-probability subspace")
            collapsed = projected / norm
            self._anchor_state = collapsed
            self._anchor_time = float(t)
            return site, collapsed.copy()
