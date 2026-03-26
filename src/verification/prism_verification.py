from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np

from src.environments.product import Policy
from src.environments.product.deterministic import DProductMDP

SPEC_LTL_MAP: dict[str, str] = {
    "gfa_gfb_g_not_h": '(G F "A") & (G F "B") & (G ! "H")',
    "f_b_g_not_h": '(F "B") & (G ! "H")',
    "f_c_f_d_gfa_gfb_g_not_h": '(F ("C" & (F ("D" & (G F "A") & (G F "B"))))) & (G ! "H")',
    "fg_e_g_not_h": '(F G "E") & (G ! "H")',
    "gf_a_or_b_then_fg_not_e": '(G F ("A" | "B")) => (F G ! "E")',
    "f_b": 'F "B"',
    "gf_b": 'G F "B"',
    "gfa_then_gf_b_or_c": '(G F "A") => (G F ("B" | "C"))',
}


class PrismVerifier:
    """Export an induced Markov chain and property files in PRISM format."""

    _ROW_SUM_TOL = 1e-12

    def __init__(
        self,
        product: DProductMDP,
        *,
        output_dir: str | Path = "tmp",
        spec_ltl_map: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(product, DProductMDP):
            raise ValueError("PrismVerifier currently supports only DProductMDP.")

        self.product = product
        self.output_dir = Path(output_dir)
        self.spec_ltl_map = dict(SPEC_LTL_MAP if spec_ltl_map is None else spec_ltl_map)

    def export_policy_mc(
        self,
        actions_map: Policy,
        spec_name: str,
        *,
        file_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Write PRISM DTMC model and property files for policy-induced MC."""
        shape = self._state_shape()
        self._validate_actions_map(actions_map, shape)
        ltl_formula = self._resolve_ltl(spec_name)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_prefix = self._slugify(file_prefix or spec_name)
        model_path = self.output_dir / f"{safe_prefix}.prism"
        props_path = self.output_dir / f"{safe_prefix}.props"

        transition_rows, transition_count = self._build_transitions(actions_map, shape)
        labels = self._build_atomic_prop_labels(shape)
        prism_label_map = {
            ap_name: self._to_prism_label_identifier(ap_name)
            for ap_name in sorted(labels.keys())
        }
        rewritten_ltl_formula = self._rewrite_ltl_formula_labels(
            ltl_formula=ltl_formula,
            prism_label_map=prism_label_map,
        )

        model_text = self._render_prism_model(
            shape=shape,
            transition_rows=transition_rows,
            labels=labels,
            prism_label_map=prism_label_map,
        )
        props_text = self._render_property_file(rewritten_ltl_formula)

        model_path.write_text(model_text, encoding="utf-8")
        props_path.write_text(props_text, encoding="utf-8")

        return {
            "model_path": str(model_path),
            "properties_path": str(props_path),
            "spec_name": spec_name,
            "ltl_formula": ltl_formula,
            "prism_ltl_formula": rewritten_ltl_formula,
            "num_states": int(np.prod(shape)),
            "num_transitions": transition_count,
            "labels": sorted(labels.keys()),
            "prism_labels": prism_label_map,
        }

    def verify_with_prism(
        self,
        actions_map: Policy,
        spec_name: str,
        *,
        prism_binary: str = "prism",
        file_prefix: str | None = None,
        extra_args: list[str] | None = None,
        timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        """Export model/properties, run PRISM, and parse satisfaction probability."""
        export_info = self.export_policy_mc(
            actions_map=actions_map,
            spec_name=spec_name,
            file_prefix=file_prefix,
        )

        command = [
            prism_binary,
            export_info["model_path"],
            export_info["properties_path"],
            *(extra_args or []),
        ]

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"PRISM binary not found: '{prism_binary}'. "
                "Install PRISM or pass a valid prism_binary path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"PRISM execution timed out after {timeout_sec} seconds."
            ) from exc

        if completed.returncode != 0:
            raise RuntimeError(
                "PRISM execution failed with non-zero exit code.\n"
                f"Command: {' '.join(command)}\n"
                f"Return code: {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        probability = self._parse_prism_probability(completed.stdout)
        if not (0.0 <= probability <= 1.0):
            raise RuntimeError(
                f"Parsed PRISM probability is out of range [0, 1]: {probability}"
            )

        return {
            **export_info,
            "satisfaction_probability": probability,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _state_shape(self) -> tuple[int, ...]:
        spec_states = int(self.product.specification.states)
        mdp_dims = tuple(
            int(x) for x in self.product.mdp.observation_space.nvec.tolist()
        )
        return (spec_states, *mdp_dims)

    def _validate_actions_map(
        self, actions_map: Policy, shape: tuple[int, ...]
    ) -> None:
        if not isinstance(actions_map, np.ndarray):
            raise ValueError("actions_map must be a numpy array.")
        if actions_map.shape != shape:
            raise ValueError(
                f"Expected actions_map of shape {shape}, got {actions_map.shape}"
            )

    def _resolve_ltl(self, spec_name: str) -> str:
        try:
            return self.spec_ltl_map[spec_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.spec_ltl_map.keys()))
            raise ValueError(
                f"Unknown spec_name '{spec_name}'. Available spec names: {available}"
            ) from exc

    def _build_transitions(
        self,
        actions_map: np.ndarray,
        shape: tuple[int, ...],
    ) -> tuple[dict[int, dict[int, float]], int]:
        transitions: dict[int, dict[int, float]] = {}
        n_transitions = 0

        for state in np.ndindex(shape):
            src = int(np.ravel_multi_index(state, shape, mode="raise"))
            action = int(actions_map[state])

            acc: dict[int, float] = {}
            for prob, succ in self.product.successors_distribution(list(state), action):
                succ_tuple = tuple(int(x) for x in succ)
                dst = int(np.ravel_multi_index(succ_tuple, shape, mode="raise"))
                acc[dst] = acc.get(dst, 0.0) + float(prob)

            total_prob = float(sum(acc.values()))
            if not np.isfinite(total_prob) or total_prob <= 0.0:
                raise ValueError(
                    f"Invalid successor distribution at state {state}: total={total_prob}"
                )
            if abs(total_prob - 1.0) > self._ROW_SUM_TOL:
                raise ValueError(
                    f"Successor probabilities at state {state} sum to {total_prob}, "
                    f"expected 1.0 +/- {self._ROW_SUM_TOL}."
                )

            transitions[src] = acc
            n_transitions += len(acc)

        return transitions, n_transitions

    def _build_atomic_prop_labels(self, shape: tuple[int, ...]) -> dict[str, set[int]]:
        labels: dict[str, set[int]] = {}

        for state in np.ndindex(shape):
            mdp_state = tuple(int(x) for x in state[1:])
            label_string = str(self.product.mdp.get_label(mdp_state))
            ap_tokens = self._parse_mdp_label_tokens(label_string)
            if not ap_tokens:
                continue

            flat_state = int(np.ravel_multi_index(state, shape, mode="raise"))
            for token in ap_tokens:
                labels.setdefault(token, set()).add(flat_state)

        return labels

    def _parse_mdp_label_tokens(self, label_string: str) -> set[str]:
        clean_label = label_string.strip()
        if not clean_label:
            return set()
        # For now we assume that each non-empty MDP label is one atomic proposition.
        return {clean_label}

    def _render_prism_model(
        self,
        *,
        shape: tuple[int, ...],
        transition_rows: dict[int, dict[int, float]],
        labels: dict[str, set[int]],
        prism_label_map: dict[str, str],
    ) -> str:
        n_states = int(np.prod(shape))
        initial_state = tuple(int(x) for x in self.product.current_state.tolist())
        initial_index = int(np.ravel_multi_index(initial_state, shape, mode="raise"))

        lines: list[str] = []
        lines.append("dtmc")
        lines.append("")
        lines.append("module induced_mc")
        lines.append(f"  s : [0..{n_states - 1}] init {initial_index};")
        for src in range(n_states):
            outgoing = transition_rows[src]
            summands = " + ".join(
                f"{prob:.16g}:(s'={dst})" for dst, prob in sorted(outgoing.items())
            )
            lines.append(f"  [] s={src} -> {summands};")
        lines.append("endmodule")

        if labels:
            lines.append("")
            for label_name in sorted(labels.keys()):
                states = sorted(labels[label_name])
                state_expr = " | ".join(f"(s={idx})" for idx in states)
                prism_name = prism_label_map[label_name]
                lines.append(f'label "{prism_name}" = {state_expr};')

        lines.append("")
        return "\n".join(lines)

    def _render_property_file(self, ltl_formula: str) -> str:
        return f"P=? [ {ltl_formula} ]\n"

    def _parse_prism_probability(self, stdout: str) -> float:
        pattern = re.compile(
            r"Result(?:\s*\(for initial states\))?:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
        )
        matches = pattern.findall(stdout)
        if not matches:
            raise RuntimeError(
                "Could not parse PRISM result probability from output.\n"
                f"STDOUT:\n{stdout}"
            )
        return float(matches[-1])

    def _slugify(self, raw_name: str) -> str:
        slug = "".join(
            ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw_name.strip()
        )
        if not slug:
            return "spec"
        return slug

    def _to_prism_label_identifier(self, raw_label: str) -> str:
        base = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw_label.strip())
        base = base.strip("_")
        if not base:
            base = "state"
        return f"ap_{base}"

    def _rewrite_ltl_formula_labels(
        self,
        *,
        ltl_formula: str,
        prism_label_map: dict[str, str],
    ) -> str:
        rewritten = ltl_formula
        for original_label, prism_label in sorted(
            prism_label_map.items(), key=lambda item: len(item[0]), reverse=True
        ):
            rewritten = rewritten.replace(f'"{original_label}"', f'"{prism_label}"')
        return rewritten
