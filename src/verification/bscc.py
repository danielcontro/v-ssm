"""Bottom strongly connected components (BSCCs) of a Markov transition matrix."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components


def bottom_bscc_labels(P: NDArray[np.floating]) -> NDArray[np.signedinteger]:
    """Labels for states in bottom SCCs of the graph with edges i -> j when P_ij > 0.

    Args:
        P: Square row-stochastic (or substochastic) transition matrix.

    Returns:
        Length-n integer array: BSCC states keep their strong-component id from
        ``connected_components``; all other states are -1.
    """
    n_components, labels = connected_components(
        P, directed=True, connection="strong", return_labels=True
    )

    comp_adj = np.zeros((n_components, n_components), dtype=bool)
    for state_idx in range(P.shape[0]):
        comp_i = labels[state_idx]
        for neighbor_idx in range(P.shape[1]):
            if P[state_idx, neighbor_idx] > 0:
                comp_j = labels[neighbor_idx]
                if comp_i != comp_j:
                    comp_adj[comp_i, comp_j] = True

    bottom_component_ids = {
        comp_id for comp_id in range(n_components) if not comp_adj[comp_id, :].any()
    }

    out = labels.astype(np.int64, copy=True)
    for state_idx in range(len(out)):
        if out[state_idx] not in bottom_component_ids:
            out[state_idx] = -1
    return out
