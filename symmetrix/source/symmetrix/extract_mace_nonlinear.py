import logging

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def _block(linear, i_in, i_out):
    for ins, w in zip(linear.instructions, linear.weight_views()):
        if (ins.i_in, ins.i_out) == (i_in, i_out):
            return (w * ins.path_weight).detach().cpu().numpy()
    raise RuntimeError(f"No path {i_in} -> {i_out} in {linear}.")


def _gate_constants(gate):
    if hasattr(gate, "_scalar_cst"):
        return gate._scalar_cst, gate._gate_cst
    return gate.act_scalars.acts[0].cst, gate.act_gates.acts[0].cst


def extract_mace_data_nonlinear(
    model, atomic_numbers, num_spline_points=256, r_min=0.5
):
    from .extract_mace_data import symmetric_contraction

    bc_type = ("not-a-knot", "clamped")
    zs = model.atomic_numbers.tolist()
    if len(atomic_numbers) == 0:
        atomic_numbers = sorted(zs)
    index = [zs.index(a) for a in atomic_numbers]
    T = len(index)
    eye = torch.eye(len(zs), dtype=torch.float64)
    if (
        len(model.interactions) != 2
        or not isinstance(model.interactions[1], type(model.interactions[0]))
        or model.spherical_harmonics._lmax != 3
    ):
        raise RuntimeError(
            "Only two-layer models of this interaction block with l_max 3 are supported."
        )
    inter_0, inter_1 = model.interactions
    prod_0, prod_1 = model.products
    scale = model.scale_shift.scale.item()
    shift = model.scale_shift.shift.item()
    E = inter_0.linear_up.irreps_out.count("0e")
    N = model.node_embedding.linear.irreps_out.count("0e")
    L_max = prod_0.linear.irreps_out.lmax
    output = {
        "format_version": 3,
        "num_channels": E,
        "num_channels_node": N,
        "ordered_pairs": True,
        "r_cut": model.r_max.item(),
        "l_max": model.spherical_harmonics._lmax,
        "L_max": L_max,
        "atomic_numbers": atomic_numbers,
        "num_elements": T,
    }
    num_LM = (L_max + 1) ** 2

    with torch.no_grad():
        H0 = model.node_embedding(eye)
        Hup0 = inter_0.linear_up(H0)
        sc0 = inter_0.skip_tp(H0)[:, :N].numpy(force=True)
        res0 = inter_0.linear_res(Hup0).numpy(force=True)
        Hup0 = Hup0.numpy(force=True)

    readout_1 = _block(model.readouts[0].linear, 0, 0)
    output["atomic_energies"] = [
        torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())[i].item()
        + shift
        + scale * float(sc0[i] @ readout_1[:, 0])
        for i in index
    ]

    from mace.modules.radial import ZBLBasis

    output["has_zbl"] = bool(getattr(model, "pair_repulsion", False))
    if output["has_zbl"]:
        zbl = model.pair_repulsion_fn
        if not isinstance(zbl, ZBLBasis):
            raise RuntimeError("Only ZBL pair_repulsion is supported.")
        output["zbl_a_exp"] = zbl.a_exp.item()
        output["zbl_a_prefactor"] = zbl.a_prefactor.item()
        output["zbl_c"] = (scale * zbl.c.numpy(force=True)).tolist()
        output["zbl_covalent_radii"] = zbl.covalent_radii.numpy(force=True).tolist()
        output["zbl_p"] = zbl.p.item()

    ### ----- RADIAL FUNCTIONS, DENSITIES AND FIRST-LAYER BASIS -----

    logging.info("radial functions")
    r = np.linspace(r_min, output["r_cut"], num_spline_points)
    r_t = torch.tensor(r, dtype=torch.float64).unsqueeze(-1)
    alpha = [inter.alpha.item() for inter in model.interactions]
    beta = [inter.beta.item() for inter in model.interactions]
    values = [[], []]
    derivs = [[], []]
    density_values = [[], []]
    density_derivs = [[], []]
    hidden = []
    for i in index:
        for j in index:
            bessels = model.radial_embedding(
                r_t, eye, torch.tensor([[j], [i]]), model.atomic_numbers
            )
            bessels, cutoff = bessels if isinstance(bessels, tuple) else (bessels, None)
            cutoff = torch.ones_like(r_t) if cutoff is None else cutoff
            for layer, inter in enumerate(model.interactions):
                with torch.no_grad():
                    feats = torch.cat(
                        [
                            bessels,
                            inter.source_embedding(eye)[[j]].expand(len(r), -1),
                            inter.target_embedding(eye)[[i]].expand(len(r), -1),
                        ],
                        dim=-1,
                    )
                    out = []
                    hook = list(inter.conv_tp_weights.modules())[
                        -1
                    ].register_forward_pre_hook(lambda m, x: out.append(x[0]))
                    R = (inter.conv_tp_weights(feats) * cutoff).numpy(force=True)
                    R *= 2 * np.sqrt(np.pi)
                    hook.remove()
                    D = (torch.tanh(inter.density_fn(feats) ** 2) * cutoff).numpy(
                        force=True
                    )[:, 0]
                if layer == 0:
                    hidden.append(
                        torch.cat([out[0] * cutoff, cutoff], dim=-1).numpy(force=True)
                    )
                spl = CubicSpline(r, R, bc_type=bc_type)
                values[layer].append(R.T.tolist())
                derivs[layer].append(spl(r, 1).T.tolist())
                spl = CubicSpline(r, D * beta[layer] / alpha[layer], bc_type=bc_type)
                density_values[layer].append(spl(r).tolist())
                density_derivs[layer].append(spl(r, 1).tolist())
    output["radial_spline_r_min"] = r_min
    for layer in range(2):
        output[f"radial_spline_values_{layer}"] = values[layer]
        output[f"radial_spline_derivs_{layer}"] = derivs[layer]
        output[f"A{layer}_scaled"] = True
        output[f"A{layer}_spline_r_min"] = r_min
        output[f"A{layer}_spline_values"] = density_values[layer]
        output[f"A{layer}_spline_derivs"] = density_derivs[layer]
    Hk, Rk = np.concatenate(hidden), np.concatenate([np.array(v).T for v in values[0]])
    U = np.linalg.lstsq(Hk, Rk, rcond=None)[0].T
    if np.linalg.norm(Hk @ U.T - Rk) > 1e-10 * np.linalg.norm(Rk):
        raise RuntimeError("the radial MLP's last layer is not linear")
    output["first_layer_weights"] = U.flatten().tolist()
    output["first_layer_values"] = [h.T.tolist() for h in hidden]
    output["first_layer_derivs"] = [
        CubicSpline(r, h, bc_type=bc_type)(r, 1).T.tolist() for h in hidden
    ]
    output["first_layer_basis_size"] = U.shape[1]

    ### ----- LAYER 0 -----

    logging.info("layer 0")
    output["H0_weights"] = Hup0[index, :].flatten().tolist()
    output["A0_weights"] = [
        [
            (_block(inter_0.linear_1, l, l) / alpha[0]).flatten().tolist()
            for l in range(4)
        ]
    ] * T
    output["residual_0"] = [res0[i, : 4 * N].tolist() for i in index]
    cst_scalar, cst_gate = _gate_constants(inter_0.equivariant_nonlin)
    output["linear_2_0"] = [
        (_block(inter_0.linear_2, l, l) * (cst_scalar if l == 0 else cst_gate))
        .flatten()
        .tolist()
        for l in range(4)
    ]

    def agnostic(module):
        return [0] * T if module.contractions[0].weights_max.shape[0] == 1 else index

    output.update(
        symmetric_contraction(
            0,
            prod_0.symmetric_contractions,
            agnostic(prod_0.symmetric_contractions),
        )
    )

    W_lin0 = [_block(prod_0.linear, l, l) for l in range(L_max + 1)]
    W_up1 = [_block(inter_1.linear_up, l, l) for l in range(L_max + 1)]
    output["H1_weights"] = np.concatenate(
        [(W_lin0[l] @ W_up1[l]).flatten() for l in range(L_max + 1)]
    ).tolist()
    output["H1_bias"] = [(sc0[i] @ W_up1[0]).tolist() for i in index]

    ### ----- LAYER 1 -----

    logging.info("Phi1")
    tp = inter_1.conv_tp
    Phi1_l = [ir[1].l for ir in tp.irreps_out]
    Phi1_clebsch_gordan, Phi1_lme, Phi1_lelm1lm2 = [], [], []

    def compute_lem(le, l, m):
        return sum(2 * Phi1_l[j] + 1 for j in range(le)) + l + m

    def compute_lme(le, l, m):
        e = le - int(sum(np.array(Phi1_l) < l))
        num_e = [int(sum(np.array(Phi1_l) == ll)) for ll in range(l + 1)]
        return sum((2 * ll + 1) * num_e[ll] for ll in range(l)) + (l + m) * num_e[l] + e

    for l1 in range(4):
        for m1 in range(-l1, l1 + 1):
            lm1 = l1 * l1 + l1 + m1
            for l2 in range(L_max + 1):
                for m2 in range(-l2, l2 + 1):
                    lm2 = l2 * l2 + l2 + m2
                    Y = torch.zeros([1, 16], dtype=torch.double)
                    Y[0, lm1] = 1.0
                    H = torch.zeros([1, num_LM * E], dtype=torch.double)
                    H[0, sum(2 * p + 1 for p in range(l2)) * E + l2 + m2] = 1.0
                    Phi = tp(
                        H,
                        Y,
                        torch.ones([1, len(tp.instructions) * E], dtype=torch.double),
                    )
                    Phi_0 = []
                    for le in range(len(tp.instructions)):
                        start = sum(2 * Phi1_l[p] + 1 for p in range(le)) * E
                        Phi_0 += [
                            Phi[0, start + p].item() for p in range(2 * Phi1_l[le] + 1)
                        ]
                    for le in range(len(tp.instructions)):
                        l = Phi1_l[le]
                        for m in range(-l, l + 1):
                            lem = compute_lem(le, l, m)
                            if np.abs(Phi_0[lem]) > 1e-12:
                                Phi1_lme.append(compute_lme(le, l, m))
                                Phi1_clebsch_gordan.append(Phi_0[lem])
                                Phi1_lelm1lm2.append((le, lm1, lm2))
    output["Phi1_l"] = Phi1_l
    output["Phi1_clebsch_gordan"] = Phi1_clebsch_gordan
    terms = list(zip(Phi1_lme, Phi1_lelm1lm2))
    Phi1_spec = sorted(set((lme, le, lm2) for lme, (le, _, lm2) in terms))
    output["Phi1_spec"] = Phi1_spec
    output["Phi1_cg_spec"] = [
        [Phi1_spec.index((lme, le, lm2)), lm1] for lme, (le, lm1, lm2) in terms
    ]

    logging.info("layer 1")
    output["A1_weights"] = [
        (
            np.vstack(
                [
                    _block(inter_1.linear_1, le, l)
                    for le in range(len(Phi1_l))
                    if Phi1_l[le] == l
                ]
            )
            / alpha[1]
        )
        .flatten()
        .tolist()
        for l in range(4)
    ]
    output["residual_1_weights"] = np.concatenate(
        [_block(inter_1.linear_res, l, l).flatten() for l in range(L_max + 1)]
    ).tolist()
    cst_scalar, cst_gate = _gate_constants(inter_1.equivariant_nonlin)
    output["linear_2_1"] = [
        (_block(inter_1.linear_2, l, l) * (cst_scalar if l == 0 else cst_gate))
        .flatten()
        .tolist()
        for l in range(4)
    ]
    output.update(
        symmetric_contraction(
            1,
            prod_1.symmetric_contractions,
            agnostic(prod_1.symmetric_contractions),
        )
    )

    ### ----- H2 AND READOUTS -----

    W_skip1 = _block(inter_1.skip_tp, 0, 0)
    output["H2_weights_for_M0"] = [(W_lin0[0] @ W_skip1).flatten().tolist()] * T
    output["H2_bias"] = [(sc0[i] @ W_skip1).tolist() for i in index]
    output["H2_weights_for_M1"] = _block(prod_1.linear, 0, 0).flatten().tolist()
    output["readout_1_weights_for_M0"] = (
        scale * (W_lin0[0] @ readout_1)[:, 0]
    ).tolist()
    output["readout_2_weights_1"] = (
        _block(model.readouts[1].linear_1, 0, 0).T.flatten().tolist()
    )
    output["readout_2_weights_2"] = (
        scale * _block(model.readouts[1].linear_2, 0, 0).T.flatten()
    ).tolist()
    act = model.readouts[1].non_linearity.acts[0]
    if act.f is not torch.nn.functional.silu:
        raise RuntimeError("Only SiLU readout activations are supported.")
    output["readout_2_scale_factor"] = act.cst
    return output
