import os
import numpy as np
from src.automata.dsa import DSA
from src.automata.specifications.maze_specs import (
    f_b_dsa,
    f_b_g_not_h_dsa,
    f_c_f_d_gfa_gfb_g_not_h_dsa,
    fg_e_g_not_h_dsa,
    gf_a_or_b_then_fg_not_e_dsa,
    gf_b_dsa,
    gfa_gfb_g_not_h_dsa,
    gfa_then_gf_b_or_c_dsa,
)
from src.environments.maze import Maze
from src.environments.product import Policy
from src.environments.product.deterministic import DProductMDP
from src.verification import DSAVerifier
from src.verification.proof_certificates import (
    BSCCsProofCertBuilder,
    ProofCertBuilder,
    StateDepDiscountProofCertBuilder,
)
from src.verification.prism_verification import SPEC_LTL_MAP


def get_safe_reachable_states(product: DProductMDP, policy: Policy):
    assert isinstance(product.mdp, Maze)
    initial_states = np.zeros(shape=product.observation_space.nvec, dtype=bool)
    initial_states[
        product.specification.initial_state, *product.mdp.initial_state.tolist()
    ] = True
    for unsafe in ["h1", "h2", "h3"]:
        unsafe_x_slice = slice(
            product.mdp.regions[unsafe].x,
            product.mdp.regions[unsafe].x + product.mdp.regions[unsafe].w,
        )
        unsafe_y_slice = slice(
            product.mdp.regions[unsafe].y,
            product.mdp.regions[unsafe].y + product.mdp.regions[unsafe].h,
        )
        initial_states[:, unsafe_x_slice, unsafe_y_slice] = False
    return product.reachable_states(policy, initial_states)


def certificate_verification(
    product: DProductMDP, policy: Policy, builder: ProofCertBuilder, log_dir: str
):
    verifier = DSAVerifier(product=product, epsilon_strict_decrease=1e-5)
    assert isinstance(product.specification, DSA)
    certificates = [
        builder.build(policy, sp) for sp in product.specification.streett_pairs
    ]

    invariant = get_safe_reachable_states(product, policy)

    os.makedirs(log_dir, exist_ok=True)

    verdict, info = verifier.verify(
        policy, invariant, certificates, log_dir=log_dir, plot=True
    )
    with open(f"{log_dir}/verification_info.txt", "w") as f:
        f.write(f"Verification verdict: {verdict}\n")
        f.write(f"Invariant violations: {info['invariant_violations']}\n")
        for i, cert_info in enumerate(info["certificates"]):
            f.write(f"Certificate {i}:\n")
            f.write(
                f"  Strict decreasing violations: {cert_info['strict_decreasing_violations']}\n"
            )
            f.write(
                f"  Non-increasing violations: {cert_info['non_increasing_violations']}\n"
            )
    np.save(f"{log_dir}/policy.npy", policy)
    product.plot_trajectories(policy, 20, 200, f"{log_dir}")

    print("\tVerification terminated:")
    if verdict:
        print("\tCertificate succesfully synthesized,")
        print("\tthe policy satisfies the specification almost surely")
    else:
        print("\tCertificate not synthesized succesfully,")
        print("\tthe policy does not satisfy the specification almost surely")

    return verdict, invariant, info


def mdp_aware_certificate_verification(
    product: DProductMDP, policy: Policy, log_dir: str
):
    discount = 0.99
    builder = BSCCsProofCertBuilder(product=product, discount=discount)
    return certificate_verification(product, policy, builder, log_dir)


def mdp_unaware_certificate_verification(
    product: DProductMDP, policy: Policy, log_dir: str
):
    discount = 0.95
    builder = StateDepDiscountProofCertBuilder(
        product=product, discount=discount, positive_reward=10.0, negative_reward=1.0
    )
    return certificate_verification(product, policy, builder, log_dir)


if __name__ == "__main__":
    parameters = {"epsilon": 1}
    specs = [
        ("f_b", f_b_dsa(parameters)),
        ("f_b_g_not_h", f_b_g_not_h_dsa(parameters)),
        ("f_c_f_d_gfa_gfb_g_not_h", f_c_f_d_gfa_gfb_g_not_h_dsa(parameters)),
        ("fg_e_g_not_h", fg_e_g_not_h_dsa(parameters)),
        ("gf_b", gf_b_dsa(parameters)),
        ("gfa_gfb_g_not_h", gfa_gfb_g_not_h_dsa(parameters)),
        ("gf_a_or_b_then_fg_not_e", gf_a_or_b_then_fg_not_e_dsa(parameters)),
        ("gfa_then_gf_b_or_c", gfa_then_gf_b_or_c_dsa(parameters)),
    ]

    mdp_name = "Maze"
    mdp = Maze(scale=2)
    for sat in ["almost_sure", "not_almost_sure"]:
        for spec_name, spec in specs:
            print(
                f"Verifying {' '.join(sat.split('_'))} satisfaction for the specification '{SPEC_LTL_MAP[spec_name]}':"
            )

            filename = f"./policies/{mdp_name}/{sat}/{spec_name}.npy"
            print(f"Loading policy {filename}")
            print()
            policy = np.load(filename)

            product = DProductMDP(mdp, spec)
            product.reset(options={"deterministic_init": True})
            mdp_aware_log_dir = f"./certificates/{mdp_name}/theorem1/{sat}/{spec_name}"
            print(
                f"\tRunning MDP-aware certificate verification, logs saved in {mdp_aware_log_dir}"
            )
            mdp_aware_certificate_verification(product, policy, mdp_aware_log_dir)
            print()
            mdp_unaware_log_dir = (
                f"./certificates/{mdp_name}/theorem2/{sat}/{spec_name}"
            )
            print(
                f"\tRunning MDP-unaware certificate verification, logs saved in {mdp_unaware_log_dir}"
            )
            mdp_unaware_certificate_verification(product, policy, mdp_unaware_log_dir)
            print()
            print()
