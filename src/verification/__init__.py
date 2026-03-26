from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.automata.dsa import DSA
from src.environments.product.deterministic import DProductMDP, ProductMDP
from src.environments.product import Policy
from src.verification.proof_certificates import ProofCertificate

type Invariant = NDArray[bool]


class Verifier(ABC):
    def __init__(
        self,
        product: ProductMDP,
        epsilon_strict_decrease: float = 1e-5,
    ):
        self.product = product
        self.epsilon_strict_decrease = epsilon_strict_decrease

    @abstractmethod
    def verify(
        self,
        actions_map: Policy,
        invariant: Invariant,
        certificates: list[ProofCertificate],
    ):
        """Check whether a policy and certificates satisfy the specification."""
        raise NotImplementedError


class DSAVerifier(Verifier):
    """Verifier implementation specialized for DSA specifications."""

    def __init__(
        self,
        product: DProductMDP,
        epsilon_strict_decrease: float = 1e-5,
    ):
        if not isinstance(product.specification, DSA):
            raise ValueError("Only DSA specifications are supported.")
        super().__init__(product, epsilon_strict_decrease)

    def verify(
        self,
        actions_map: Policy,
        invariant: Invariant,
        certificates: list[ProofCertificate],
        log_dir: str | None = None,
        plot: bool = False,
    ):
        if not isinstance(actions_map, np.ndarray):
            raise ValueError("actions_map must be a numpy array.")
        spec_states = int(self.product.specification.states)
        mdp_dims = tuple(
            int(x) for x in self.product.mdp.observation_space.nvec.tolist()
        )
        shape = (spec_states, *mdp_dims)
        if actions_map.shape != shape:
            raise ValueError(f"actions_map must have shape {shape}")
        if invariant.shape != shape:
            raise ValueError(f"invariant must have shape {shape}")
        if len(certificates) != len(self.product.specification.streett_pairs):
            raise ValueError(
                f"certificate must have length {len(self.product.specification.streett_pairs)}"
            )
        for i, cert in enumerate(certificates):
            if cert.shape != shape:
                raise ValueError(f"certificates[{i}] must have shape {shape}")

        info_certificates = []
        verdict = True
        for i, (fin_set, inf_set) in enumerate(
            self.product.specification.streett_pairs
        ):
            fin_mask = np.zeros(shape, dtype=bool)
            fin_indices = sorted(fin_set)
            fin_mask[fin_indices, ...] = True

            inf_mask = np.zeros(shape, dtype=bool)
            inf_indices = sorted(inf_set)
            inf_mask[inf_indices, ...] = True

            certificate = certificates[i]
            certificate_expected_next_state = self._certificate_expected_next_state(
                actions_map, certificate
            )
            I_cap_A_cap_compl_B = np.logical_and(
                invariant, np.logical_and(fin_mask, ~inf_mask)
            )
            certificate_strict_decreasing = self._certificate_strict_decreasing(
                certificate, certificate_expected_next_state, I_cap_A_cap_compl_B
            )
            I_cap_compl__A_cup_B = np.logical_and(
                invariant, ~np.logical_or(fin_mask, inf_mask)
            )
            certificate_non_increasing = self._certificate_non_increasing(
                certificate, certificate_expected_next_state, I_cap_compl__A_cup_B
            )

            info_certificates.append(
                {
                    "strict_decreasing_violations": np.sum(
                        ~certificate_strict_decreasing
                    ),
                    "non_increasing_violations": np.sum(~certificate_non_increasing),
                }
            )

            if plot:
                self.product.plot_heatmap(
                    certificate,
                    actions_map,
                    save_dir=log_dir,
                    filename=f"certificate_{i}.svg",
                    title=f"Certificate {i}",
                    invariant_states=invariant,
                )
                diff_matrix = certificate_expected_next_state - certificate
                self.product.plot_heatmap(
                    diff_matrix,
                    actions_map,
                    save_dir=log_dir,
                    filename=f"diff_certificate_{i}.svg",
                    title=f"PostV-V difference for certificate {i}",
                    invariant_states=invariant,
                )

                # Plot PostV-V+eps for strict epsilon decrease states
                self.product.plot_heatmap(
                    diff_matrix + self.epsilon_strict_decrease,
                    actions_map,
                    save_dir=log_dir,
                    filename=f"sign_eps_decr_diff_certificate_{i}.svg",
                    title="sign PostV-V+eps",
                    sign=True,
                    invariant_states=I_cap_A_cap_compl_B,
                )
                self.product.plot_heatmap(
                    diff_matrix,
                    actions_map,
                    save_dir=log_dir,
                    filename=f"sign_non_incr_diff_certificate_{i}.svg",
                    title="sign PostV-V",
                    sign=True,
                    invariant_states=I_cap_compl__A_cup_B,
                )

            verdict = (
                verdict
                and np.all(certificate_strict_decreasing)
                and np.all(certificate_non_increasing)
            )

        invariant_satisfaction = self._invariant_satisfaction(actions_map, invariant)
        verdict = bool(verdict and np.all(invariant_satisfaction))
        info = {
            "invariant_violations": np.sum(~invariant_satisfaction),
            "certificates": info_certificates,
        }

        return verdict, info

    def _invariant_satisfaction(
        self, actions_map: Policy, invariant: Invariant
    ) -> NDArray[bool]:
        sat_matrix = np.ones(actions_map.shape, dtype=bool)
        for s in np.ndindex(actions_map.shape):
            if not invariant[s]:
                continue
            a = int(actions_map[s])
            for prob, succ in self.product.successors_distribution(list(s), a):
                succ_idx = tuple(int(x) for x in succ)
                if prob > 0 and not invariant[succ_idx]:
                    sat_matrix[s] = False
                    break
        return sat_matrix

    def _certificate_strict_decreasing(
        self,
        certificate: ProofCertificate,
        certificate_expected_next_state: ProofCertificate,
        checkeable_states: NDArray[bool],
    ) -> NDArray[bool]:
        difference_matrix = certificate - certificate_expected_next_state
        return np.logical_or(
            difference_matrix >= self.epsilon_strict_decrease, ~checkeable_states
        )

    def _certificate_non_increasing(
        self,
        certificate: ProofCertificate,
        certificate_expected_next_state: ProofCertificate,
        checkeable_states: NDArray[bool],
        tol: float = 0,
    ) -> NDArray[bool]:
        difference_matrix = certificate - certificate_expected_next_state
        return np.logical_or(difference_matrix >= -tol, ~checkeable_states)

    def _certificate_expected_next_state(
        self, actions_map: Policy, certificate: ProofCertificate
    ) -> ProofCertificate:
        certificate_expected_next_state = np.zeros_like(certificate)
        for s in np.ndindex(certificate.shape):
            a = int(actions_map[s])
            for prob, succ in self.product.successors_distribution(list(s), a):
                if prob > 0:
                    succ_idx = tuple(int(x) for x in succ)
                    certificate_expected_next_state[s] += prob * certificate[succ_idx]
        return certificate_expected_next_state


class QTableVerifier(Verifier):
    pass


class QNetVerifier(Verifier):
    pass
