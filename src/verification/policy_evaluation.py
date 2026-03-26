import sys
import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Optional

from src.environments.product import Policy
from src.environments.product.deterministic import DProductMDP
from src.verification.bscc import bottom_bscc_labels

ProductState = Sequence[int]  # e.g. [q, *mdp_state]


def _as_tuple_state(state: ProductState) -> tuple[int, ...]:
    return tuple(int(x) for x in state)


def compute_value_function_iterative(
    product: DProductMDP,
    actions_map: Policy,
    discount: float,
    *,
    tol: float = 1e-6,
    max_iters: int = 10000,
    reward_matrix: Optional[NDArray] = None,
    state_dependent_discount: bool = False,
) -> NDArray:
    """Policy evaluation on a DProductMDP using full transition knowledge.

    Computes the infinite-horizon discounted value function V^π satisfying:
      V(s) = r(s) + discount * Σ_{s'} P(s'|s, π(s)) V(s')
    If state_dependent_discount is True, uses:
      V(s) = r(s) + 1.0 * E[V(s')]   when r(s) == 0
      V(s) = r(s) + discount * E[V(s')] when r(s) != 0

    Returns an array V with shape (spec_states, *mdp_dims).
    """
    if not (0.0 <= discount < 1.0):
        raise ValueError("discount must be in [0, 1).")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0.")
    spec_states = int(product.specification.states)
    mdp_dims = tuple(int(x) for x in product.mdp.observation_space.nvec.tolist())
    shape = (spec_states, *mdp_dims)
    if not isinstance(actions_map, np.ndarray):
        raise ValueError("actions_map must be a numpy array.")
    if reward_matrix is not None and reward_matrix.shape != shape:
        raise ValueError(f"reward_matrix must have shape {shape}")
    if actions_map.shape != shape:
        raise ValueError(f"actions_map must have shape {shape}")

    if reward_matrix is None:
        # Reward depends only on the specification state q (first component).
        r_by_q = np.array(
            [float(product.specification.get_reward(q)) for q in range(spec_states)],
            dtype=float,
        )
        R = np.broadcast_to(
            r_by_q.reshape((spec_states,) + (1,) * len(mdp_dims)), shape
        )
    else:
        R = reward_matrix

    V = np.zeros(shape, dtype=float)

    for _ in range(max_iters):
        V_new = np.empty_like(V)

        for s in np.ndindex(shape):
            a = int(actions_map[s])
            exp_next = 0.0
            for prob, succ in product.successors_distribution(list(s), a):
                exp_next += float(prob) * float(V[_as_tuple_state(succ)])
            reward = float(R[s])
            if state_dependent_discount:
                gamma = 1.0 if reward == 0.0 else discount
            else:
                gamma = discount
            V_new[s] = reward + gamma * exp_next

        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new

    return V


def compute_value_function_linear_solver(
    product: DProductMDP,
    actions_map: Policy,
    discount: float,
    *,
    reward_matrix: Optional[NDArray] = None,
    state_dependent_discount: bool = False,
) -> NDArray:
    """Policy evaluation by solving the Bellman equation (I - γ P_π) V = R exactly.

    Same V^π as compute_value_function_iterative, but via one linear solve instead of iteration.
    Use when the state space is small enough that building the transition matrix is feasible.
    If state_dependent_discount is True, uses γ(s)=1.0 when R(s)==0 and γ(s)=discount otherwise.
    In that mode, on each bottom SCC where every state has R=0, rows are replaced by V_i=0 so
    the system is nonsingular; this matches value iteration started from V_0=0 on those traps.
    """
    if not (0.0 <= discount < 1.0):
        raise ValueError("discount must be in [0, 1).")
    spec_states = int(product.specification.states)
    mdp_dims = tuple(int(x) for x in product.mdp.observation_space.nvec.tolist())
    shape = (spec_states, *mdp_dims)
    if not isinstance(actions_map, np.ndarray):
        raise ValueError("actions_map must be a numpy array.")
    if reward_matrix is not None and reward_matrix.shape != shape:
        raise ValueError(f"reward_matrix must have shape {shape}")
    if actions_map.shape != shape:
        raise ValueError(f"actions_map must have shape {shape}")

    n = int(np.prod(shape))
    P = np.zeros((n, n))
    R = np.zeros(n)
    for i, s in enumerate(np.ndindex(shape)):
        if reward_matrix is None:
            R[i] = float(product.specification.get_reward(s[0]))
        else:
            R[i] = float(reward_matrix[s])
        a = int(actions_map[s])
        for prob, succ in product.successors_distribution(list(s), a):
            j = np.ravel_multi_index(tuple(int(x) for x in succ), shape, mode="raise")
            P[i, j] += float(prob)
    if state_dependent_discount:
        gamma = np.where(R == 0.0, 1.0, discount)
    else:
        gamma = np.full(n, discount, dtype=float)
    A = np.eye(n) - gamma[:, None] * P
    if state_dependent_discount:
        bscc_labels = bottom_bscc_labels(P)
        for comp_id in np.unique(bscc_labels):
            if comp_id < 0:
                continue
            idxs = np.flatnonzero(bscc_labels == comp_id)
            if idxs.size == 0 or not np.all(R[idxs] == 0.0):
                continue
            A[idxs, :] = 0.0
            A[idxs, idxs] = 1.0
            R[idxs] = 0.0
    try:
        V_flat = np.linalg.solve(A, R)
    except np.linalg.LinAlgError:
        # When gamma(s)=1 on zero-reward recurrent classes, A = I - γP can be singular.
        print(
            "compute_value_function_linear_solver: np.linalg.solve failed; "
            "dumping Bellman system A V = R for diagnosis.\n",
            file=sys.stderr,
        )
        print(f"n_states={n}, rank(A)={np.linalg.matrix_rank(A)}", file=sys.stderr)
        ru, rc = np.unique(R, return_counts=True)
        print(
            "R value_counts:",
            {float(v): int(c) for v, c in zip(ru, rc)},
            file=sys.stderr,
        )
        print(
            f"n_states with R==0 (use gamma=1 when state_dependent_discount): "
            f"{int(np.sum(R == 0.0))}",
            file=sys.stderr,
        )
        gu, gc = np.unique(gamma, return_counts=True)
        print(
            "gamma value_counts:",
            {float(v): int(c) for v, c in zip(gu, gc)},
            file=sys.stderr,
        )
        with np.printoptions(precision=6, suppress=True, linewidth=200):
            print("gamma =", gamma, file=sys.stderr)
            print("R =", R, file=sys.stderr)
            print("A =\n", A, file=sys.stderr)
            if n <= 64:
                w = np.linalg.eigvals(A)
                print("eigvals(A) =", w, file=sys.stderr)
            try:
                s = np.linalg.svd(A, compute_uv=False)
                print("singular values (desc):", s, file=sys.stderr)
            except np.linalg.LinAlgError:
                print("SVD of A also failed.", file=sys.stderr)
        raise
    return V_flat.reshape(shape)
