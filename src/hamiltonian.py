from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


COMPLEX = np.complex128
I2 = np.eye(2, dtype=COMPLEX)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX)
LOWERING = np.array([[0, 1], [0, 0]], dtype=COMPLEX)  # |0><1|
RAISING = LOWERING.conj().T  # |1><0|
P0 = np.array([[1, 0], [0, 0]], dtype=COMPLEX)
P1 = np.array([[0, 0], [0, 1]], dtype=COMPLEX)


def sqrt_not() -> np.ndarray:
    """Principal square-root of NOT: U^2 = X."""
    return 0.5 * np.array(
        [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=COMPLEX
    )


def gate_from_name(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key in {"sqrt_not", "sqrt-not", "rnot", "sqrtx"}:
        return sqrt_not()
    if key in {"not", "x"}:
        return X.copy()
    if key in {"identity", "i", "id"}:
        return I2.copy()
    raise ValueError(f"Unsupported gate name: {name!r}")


def kron_all(ops: Iterable[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]], dtype=COMPLEX)
    for op in ops:
        result = np.kron(result, op)
    return result


def embedded_operator(num_qubits: int, operators: dict[int, np.ndarray]) -> np.ndarray:
    ops = [operators.get(i, I2) for i in range(num_qubits)]
    return kron_all(ops)


def basis_state(bits: list[int]) -> np.ndarray:
    vec = np.array([1.0 + 0.0j], dtype=COMPLEX)
    for bit in bits:
        local = np.array([1.0, 0.0], dtype=COMPLEX) if bit == 0 else np.array([0.0, 1.0], dtype=COMPLEX)
        vec = np.kron(vec, local)
    return vec


@dataclass(frozen=True)
class FeynmanClockModel:
    cursor_sites: int
    gate_names: tuple[str, ...]
    coupling: float = 1.0
    hbar: float = 1.0
    answer_initial: int = 0

    def __post_init__(self) -> None:
        if self.cursor_sites < 2:
            raise ValueError("cursor_sites must be >= 2")
        if len(self.gate_names) != self.cursor_sites - 1:
            raise ValueError("Need exactly cursor_sites - 1 gates")
        if self.answer_initial not in (0, 1):
            raise ValueError("answer_initial must be 0 or 1")
        if self.hbar <= 0:
            raise ValueError("hbar must be positive")

    @property
    def num_qubits(self) -> int:
        return self.cursor_sites + 1

    @property
    def answer_qubit(self) -> int:
        return self.cursor_sites

    @property
    def hilbert_dim(self) -> int:
        return 2 ** self.num_qubits

    @property
    def gates(self) -> list[np.ndarray]:
        return [gate_from_name(name) for name in self.gate_names]

    def initial_state(self) -> np.ndarray:
        # PCS0 occupied, all other PCS sites unoccupied, answer in configured basis state.
        bits = [1] + [0] * (self.cursor_sites - 1) + [self.answer_initial]
        return basis_state(bits)

    def cursor_projector(self, site: int) -> np.ndarray:
        if not 0 <= site < self.cursor_sites:
            raise IndexError(site)
        return embedded_operator(self.num_qubits, {site: P1})

    def hamiltonian(self) -> np.ndarray:
        """
        Feynman clock Hamiltonian

            H = J Σ_i [ q†_(i+1) q_i A_(i+1) + h.c. ]

        where A acts on the answer qubit.  The h.c. term makes H Hermitian and
        implements the reversible backward transition with A†.
        """
        H = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=COMPLEX)
        for i, gate in enumerate(self.gates):
            forward = embedded_operator(
                self.num_qubits,
                {
                    i: LOWERING,
                    i + 1: RAISING,
                    self.answer_qubit: gate,
                },
            )
            H += self.coupling * (forward + forward.conj().T)
        return H

    def expected_answer_state_at_site(self, site: int) -> np.ndarray:
        """Ideal conditional answer ket after the first `site` gates."""
        ket = np.array([1.0, 0.0], dtype=COMPLEX) if self.answer_initial == 0 else np.array([0.0, 1.0], dtype=COMPLEX)
        for gate in self.gates[:site]:
            ket = gate @ ket
        return ket / np.linalg.norm(ket)
