import numpy as np
from argparse import ArgumentParser
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
from src.verification.prism_verification import SPEC_LTL_MAP, PrismVerifier


def prism_verification(
    product: DProductMDP,
    actions_map: Policy,
    spec_name: str,
    prism_binary: str,
    timeout_sec: int,
):
    assert isinstance(product.specification, DSA)
    verifier = PrismVerifier(product)
    result = verifier.verify_with_prism(
        actions_map,
        spec_name,
        prism_binary=prism_binary,
        timeout_sec=timeout_sec,
    )
    print(
        f"Satisfaction probability of '{SPEC_LTL_MAP[spec_name]}': {result['satisfaction_probability']}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script for computing satisfaction probabilities of policies with PRISM."
    )

    parser.add_argument("--prism_binary", type=str, help="Path to the PRISM binary")

    parser.add_argument(
        "--timeout",
        type=int,
        default=200,
        help="Timeout(s) for PRISM verification",
    )

    args = parser.parse_args()

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
    maze = Maze(scale=2)
    for sat in ["almost_sure", "not_almost_sure"]:
        for spec_name, spec in specs:
            print(
                f"Computing satisfaction probability of the specification '{SPEC_LTL_MAP[spec_name]}':"
            )
            policy_filename = f"./policies/{mdp_name}/{sat}/{spec_name}.npy"
            print(f"Loading policy {policy_filename}")
            policy = np.load(policy_filename)
            product = DProductMDP(maze, spec)
            product.reset(options={"deterministic_init": True})
            prism_verification(
                product, policy, spec_name, args.prism_binary, args.timeout
            )
            print()
            print()
