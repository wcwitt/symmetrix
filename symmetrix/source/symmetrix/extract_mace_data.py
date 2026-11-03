import torch

import logging
import numpy as np
from scipy.interpolate import CubicSpline

with torch.serialization.safe_globals([slice]):
    from e3nn.o3 import Irreps, Linear, wigner_3j
from mace.modules.radial import ZBLBasis
from mace.tools.scripts_utils import remove_pt_head

from ase.data import chemical_symbols


def extract_mace_data(
    model,
    species,
    head=None,
    num_spline_points=1024,
    r_min=0.5,
    compression_margin=None,
):
    """Extract data from pytorch model file into structure that can be
    written as symmetrix JSON data file

    Parameters
    ----------
    model: str / Path
        path to pytorch model file
    species: list(int / str)
        list of atomic numbers or chemical symbols to extract
    head: str, default None
        head to keep, if multihead model, default same as mace.tools.scripts_utils.remove_pt_head
    num_spline_points: int, default 1024
        number of spline points to approximate various functions
    r_min: float, default 0.5
        minimum distance covered by the splines
    compression_margin: float, default None
        compress the first-layer radial basis to 32 functions per ordered pair if
        its error stays below this fraction of the spline error; off by default
        because the compression error is systematic and shifts the pressure

    Returns
    -------
    output_data: dict with symmetrix model data
    """
    device = torch.device("cpu")
    model = torch.load(model, map_location=device, weights_only=False).to(
        device=device, dtype=torch.float64
    )
    model.eval()

    if species is None:
        species = []

    # extract atomic numbers
    atomic_numbers = []
    for sp in species:
        try:
            Z = int(sp)
        except ValueError:
            try:
                Z = chemical_symbols.index(sp)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse {sp} as atomic number or chemical species"
                ) from exc
        atomic_numbers.append(Z)

    # ensure that splines goes smoothly to 0 at outer cutoff
    spline_bc_type = ("not-a-knot", "clamped")

    ### ----- EXTRACT SINGLE HEAD -----

    if hasattr(model, "heads") and len(model.heads) != 1:
        torch.set_default_dtype(next(model.parameters()).dtype)
        model = remove_pt_head(model, head).to(device=device, dtype=torch.float64)
        model.eval()

    ### ----- CHECK FOR COMPATIBILITY -----

    if (
        type(model.interactions[0]).__name__
        == "RealAgnosticResidualNonLinearInteractionBlock"
    ):
        from .extract_mace_nonlinear import extract_mace_data_nonlinear

        return extract_mace_data_nonlinear(
            model, atomic_numbers, num_spline_points, r_min
        )

    if len(model.interactions) != 2:
        raise RuntimeError("Currently, symmetrix only supports two-layer MACE models.")

    from mace.modules.blocks import (
        RealAgnosticInteractionBlock,
        RealAgnosticDensityInteractionBlock,
        RealAgnosticResidualInteractionBlock,
        RealAgnosticDensityResidualInteractionBlock,
    )

    first_residual = isinstance(
        model.interactions[0], RealAgnosticResidualInteractionBlock
    )
    if (
        not isinstance(model.interactions[0], RealAgnosticInteractionBlock)
        and not isinstance(model.interactions[0], RealAgnosticDensityInteractionBlock)
        and not first_residual
    ):
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose first interaction is "
            "RealAgnosticInteractionBlock, RealAgnosticDensityInteractionBlock or RealAgnosticResidualInteractionBlock."
        )

    if not isinstance(
        model.interactions[1], RealAgnosticResidualInteractionBlock
    ) and not isinstance(
        model.interactions[1], RealAgnosticDensityResidualInteractionBlock
    ):
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose second interaction is "
            "RealAgnosticResidualInteractionBlock or RealAgnosticDensityResidualInteractionBlock."
        )

    if model.spherical_harmonics._lmax != 3:
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models with l_max=3."
        )

    ### ----- HELPER FUNCTION -----

    def linear_simplify(linear):
        simplified = Linear(
            Irreps(linear.irreps_in).simplify(), Irreps(linear.irreps_out).simplify()
        )
        simplified.weight = linear.weight
        simplified.bias = linear.bias
        return simplified.to(device=device, dtype=torch.float64)

    ### ----- BASIC MODEL INFO -----

    num_channels = model.node_embedding.linear.irreps_out.count("0e")
    r_cut = model.r_max.item()
    l_max = model.spherical_harmonics._lmax
    L_max = model.products[0].linear.irreps_out.lmax
    output = {"format_version": 2}
    output["num_channels"] = num_channels
    output["r_cut"] = r_cut
    output["l_max"] = l_max
    output["L_max"] = L_max

    ### ----- ATOMIC NUMBERS AND ENERGIES -----

    atomic_numbers = sorted(atomic_numbers)
    if len(atomic_numbers) == 0:
        atomic_numbers = sorted(model.atomic_numbers.tolist())
        logging.warning(f"No atomic_numbers, including all: {atomic_numbers}")
    atomic_energies = [
        torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())[
            model.atomic_numbers.tolist().index(a)
        ].item()
        + model.scale_shift.shift.item()
        for a in atomic_numbers
    ]
    output["atomic_numbers"] = atomic_numbers
    output["num_elements"] = len(atomic_numbers)
    output["atomic_energies"] = atomic_energies

    ### --- ZBL ---

    if hasattr(model, "pair_repulsion") and model.pair_repulsion:
        if not isinstance(model.pair_repulsion_fn, ZBLBasis):
            raise Exception("Only ZBL pair_repulsion is supported.")
        output["has_zbl"] = True
        zbl = model.pair_repulsion_fn
        output["zbl_a_exp"] = zbl.a_exp.item()
        output["zbl_a_prefactor"] = zbl.a_prefactor.item()
        output["zbl_c"] = (
            model.scale_shift.scale.item() * zbl.c.numpy(force=True)
        ).tolist()
        output["zbl_covalent_radii"] = zbl.covalent_radii.numpy(force=True).tolist()
        output["zbl_p"] = zbl.p.item()
    else:
        output["has_zbl"] = False

    ### ----- RADIAL SPLINES -----

    logging.info("R0+R1")
    r = np.linspace(r_min, r_cut, num_spline_points)
    spline_values_0 = []
    spline_derivatives_0 = []
    spline_values_1 = []
    spline_derivatives_1 = []
    for a_i in atomic_numbers:
        for a_j in atomic_numbers:
            if a_j < a_i:
                continue
            model_i = model.atomic_numbers.tolist().index(a_i)
            model_j = model.atomic_numbers.tolist().index(a_j)
            bessels = model.radial_embedding(
                torch.tensor(r, dtype=torch.float64, device=device).unsqueeze(-1),
                torch.eye(len(model.atomic_numbers), device=device),
                torch.tensor([[model_i], [model_j]], dtype=torch.int64, device=device),
                model.atomic_numbers.to(device),
            )
            bessels, cutoff = bessels if isinstance(bessels, tuple) else (bessels, None)
            cutoff = 1 if cutoff is None else cutoff
            # radial basis for interaction 0
            R = (model.interactions[0].conv_tp_weights(bessels) * cutoff).numpy(
                force=True
            )
            spl = CubicSpline(r, 2 * np.sqrt(np.pi) * R, axis=0, bc_type=spline_bc_type)
            spline_values_0.append(spl(r).T.tolist())
            spline_derivatives_0.append(spl(r, 1).T.tolist())
            # radial basis for interaction 1
            R = (model.interactions[1].conv_tp_weights(bessels) * cutoff).numpy(
                force=True
            )
            spl = CubicSpline(r, 2 * np.sqrt(np.pi) * R, axis=0, bc_type=spline_bc_type)
            spline_values_1.append(spl(r).T.tolist())
            spline_derivatives_1.append(spl(r, 1).T.tolist())

    output["radial_spline_r_min"] = float(r[0])
    output["radial_spline_values_0"] = spline_values_0
    output["radial_spline_derivs_0"] = spline_derivatives_0
    output["radial_spline_values_1"] = spline_values_1
    output["radial_spline_derivs_1"] = spline_derivatives_1

    ### ----- H0 -----

    logging.info("H0")
    H0_weights = (
        np.reshape(
            model.node_embedding.linear.weight.numpy(force=True),
            [len(model.atomic_numbers), num_channels],
        )
        / np.sqrt(len(model.atomic_numbers))
        @ np.reshape(
            model.interactions[0].linear_up.weight.numpy(force=True),
            [num_channels, num_channels],
        )
        / np.sqrt(num_channels)
    )
    indices = [model.atomic_numbers.tolist().index(a) for a in atomic_numbers]
    H0_weights = H0_weights[indices, :]
    output["H0_weights"] = H0_weights.flatten().tolist()

    ### ----- Phi0 -----

    logging.info("Phi0")

    ### ----- A0 -----

    logging.info("A0")
    A0_scaled = True if ("Density" in type(model.interactions[0]).__name__) else False
    output["A0_scaled"] = A0_scaled
    if A0_scaled:
        r = np.linspace(r_min, r_cut, num_spline_points)
        A0_spline_values = []
        A0_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=torch.float64, device=device).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers), device=device),
                    torch.tensor(
                        [[model_i], [model_j]], dtype=torch.int64, device=device
                    ),
                    model.atomic_numbers.to(device),
                )
                bessels, cutoff = (
                    bessels if isinstance(bessels, tuple) else (bessels, None)
                )
                cutoff = 1 if cutoff is None else cutoff
                R = (
                    torch.tanh(model.interactions[0].density_fn(bessels) ** 2) * cutoff
                ).numpy(force=True)
                spl = CubicSpline(r, R[:, 0], bc_type=spline_bc_type)
                A0_spline_values.append(spl(r).tolist())
                A0_spline_derivs.append(spl.derivative()(r).tolist())
        output["A0_spline_r_min"] = float(r[0])
        output["A0_spline_values"] = A0_spline_values
        output["A0_spline_derivs"] = A0_spline_derivs
    A0_weights = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        A0_weights.append([])
        for l, _, w in (
            model.interactions[0].skip_tp.weight_views(yield_instruction=True)
            if not first_residual
            else [(l, None, None) for l in range(l_max + 1)]
        ):
            w_linear = model.interactions[0].linear.weight_view_for_instruction(
                l
            ).numpy(force=True) / np.sqrt(num_channels)
            if not A0_scaled:
                w_linear /= model.interactions[0].avg_num_neighbors
            fused = (
                w_linear
                if w is None
                else (
                    w_linear
                    @ w[:, model_i, :].numpy(force=True)
                    / np.sqrt(len(model.atomic_numbers) * num_channels)
                )
            )
            A0_weights[i].append(fused.flatten().tolist())
    output["A0_weights"] = A0_weights

    ### ----- FIRST-LAYER BASIS -----

    logging.info("first-layer basis")
    output.update(
        first_layer_basis(
            model,
            atomic_numbers,
            output,
            num_spline_points,
            r_min,
            compression_margin,
            spline_bc_type,
        )
    )

    #### ----- M0 -----

    logging.info("M0")
    elements = [model.atomic_numbers.tolist().index(a) for a in atomic_numbers]
    output.update(
        symmetric_contraction(0, model.products[0].symmetric_contractions, elements)
    )

    ### ----- H1 -----

    logging.info("H1")
    H1_weights = np.zeros([L_max + 1, num_channels, num_channels])
    weights_0 = np.reshape(
        model.products[0].linear.weight.numpy(force=True),
        [L_max + 1, num_channels, num_channels],
    ) / np.sqrt(num_channels)
    weights_1 = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True),
        [L_max + 1, num_channels, num_channels],
    ) / np.sqrt(num_channels)
    for l in range(L_max + 1):
        H1_weights[l, :, :] = weights_0[l, :, :] @ weights_1[l, :, :]
    output["H1_weights"] = H1_weights.flatten().tolist()
    if first_residual:
        with torch.no_grad():
            eye = torch.eye(len(model.atomic_numbers), dtype=torch.float64)
            sc0 = (
                model.interactions[0]
                .skip_tp(model.node_embedding(eye), eye)[:, :num_channels]
                .numpy(force=True)
            )
        output["H1_bias"] = [(sc0[i] @ weights_1[0]).tolist() for i in indices]
        W_skip = model.interactions[1].skip_tp.weight_view_for_instruction(0)
        output["H2_bias"] = [
            (
                sc0[i]
                @ W_skip[:, i, :].numpy(force=True)
                / np.sqrt(len(model.atomic_numbers) * num_channels)
            ).tolist()
            for i in indices
        ]
        readout_1 = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(
            num_channels
        )
        output["atomic_energies"] = [
            e + model.scale_shift.scale.item() * float(sc0[i] @ readout_1)
            for e, i in zip(output["atomic_energies"], indices)
        ]

    ### ----- Phi1 -----

    logging.info("Phi1")
    Phi1_l = [ir[1].l for ir in model.interactions[1].conv_tp.irreps_out]
    Phi1_clebsch_gordan = []
    Phi1_lme = []
    Phi1_lelm1lm2 = []
    num_lm1 = (l_max + 1) ** 2
    num_lm2 = (L_max + 1) ** 2

    def compute_lem(le, l, m):
        lem = 0
        for j in range(le):
            lem += 2 * Phi1_l[j] + 1
        return lem + l + m

    def compute_lme(le, l, m):
        e = le - int(sum(np.array(Phi1_l) < l))
        num_e = [int(sum(np.array(Phi1_l) == ll)) for ll in range(l + 1)]
        lme = 0
        for ll in range(l):
            lme += (2 * ll + 1) * num_e[ll]
        return lme + (l + m) * num_e[l] + e

    tp = model.interactions[1].conv_tp.to(device)
    for l1 in range(l_max + 1):
        for m1 in range(-l1, l1 + 1):
            lm1 = l1 * l1 + l1 + m1
            for l2 in range(L_max + 1):
                for m2 in range(-l2, l2 + 1):
                    lm2 = l2 * l2 + l2 + m2
                    R = torch.ones(
                        [1, len(tp.instructions) * num_channels],
                        dtype=torch.double,
                        device=device,
                    )
                    Y = torch.zeros([1, num_lm1], dtype=torch.double, device=device)
                    Y[0, lm1] = 1.0
                    H = torch.zeros(
                        [1, num_lm2 * num_channels], dtype=torch.double, device=device
                    )
                    H[
                        0, sum([2 * p + 1 for p in range(l2)]) * num_channels + l2 + m2
                    ] = 1.0
                    Phi = tp(H, Y, R)
                    # extract Phi values for k=0
                    Phi_0 = []
                    for le in range(len(tp.instructions)):
                        Phi_0_start = (
                            sum([2 * Phi1_l[p] + 1 for p in range(le)]) * num_channels
                        )
                        for p in range(2 * Phi1_l[le] + 1):
                            Phi_0.append(Phi[0, Phi_0_start + p].item())
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
    # Phi1[lme] += CG_Y[q] R1[lel1l2] H1[lm2], with CG_Y[q] += CG Y[lm1]
    terms = list(zip(Phi1_lme, Phi1_lelm1lm2))
    Phi1_spec = sorted(set((lme, le, lm2) for lme, (le, _, lm2) in terms))
    output["Phi1_spec"] = Phi1_spec
    output["Phi1_cg_spec"] = [
        [Phi1_spec.index((lme, le, lm2)), lm1] for lme, (le, lm1, lm2) in terms
    ]

    ### ----- A1 -----

    logging.info("A1")
    A1_scaled = True if ("Density" in type(model.interactions[1]).__name__) else False
    output["A1_scaled"] = A1_scaled
    if A1_scaled:
        r = np.linspace(r_min, r_cut, num_spline_points)
        A1_spline_values = []
        A1_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=torch.float64, device=device).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers), device=device),
                    torch.tensor(
                        [[model_i], [model_j]], dtype=torch.int64, device=device
                    ),
                    model.atomic_numbers.to(device),
                )
                bessels, cutoff = (
                    bessels if isinstance(bessels, tuple) else (bessels, None)
                )
                cutoff = 1 if cutoff is None else cutoff
                R = (
                    torch.tanh(model.interactions[1].density_fn(bessels) ** 2) * cutoff
                ).numpy(force=True)
                spl = CubicSpline(r, R[:, 0], bc_type=spline_bc_type)
                A1_spline_values.append(spl(r).tolist())
                A1_spline_derivs.append(spl.derivative()(r).tolist())
        output["A1_spline_r_min"] = float(r[0])
        output["A1_spline_values"] = A1_spline_values
        output["A1_spline_derivs"] = A1_spline_derivs
    A1_weights = []
    num_eta = [sum([l == ll for ll in Phi1_l]) for l in range(l_max + 1)]
    A1_linear = linear_simplify(model.interactions[1].linear)
    for l in range(l_max + 1):
        w_linear = A1_linear.weight_view_for_instruction(l).numpy(force=True) / np.sqrt(
            num_eta[l] * num_channels
        )
        if not A1_scaled:
            w_linear /= model.interactions[1].avg_num_neighbors
        w_linear = np.reshape(w_linear, (num_eta[l], num_channels, num_channels))
        A1_weights.append(w_linear.flatten().tolist())
    output["A1_weights"] = A1_weights

    ### ----- M1 -----

    logging.info("M1")
    output.update(
        symmetric_contraction(1, model.products[1].symmetric_contractions, elements)
    )

    ### ----- H2 -----

    logging.info("H2")
    # H2 weights for M0
    H2_weights_for_M0 = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        w = (
            model.interactions[1]
            .skip_tp.weight_view_for_instruction(0)[:, model_i, :]
            .numpy(force=True)
        )
        H2_weights_for_M0.append(
            (weights_0[0] @ w / np.sqrt(len(model.atomic_numbers) * num_channels))
            .flatten()
            .tolist()
        )
    output["H2_weights_for_M0"] = H2_weights_for_M0
    # H2 weights for M1
    output["H2_weights_for_M1"] = (
        model.products[1].linear.weight.numpy(force=True) / np.sqrt(num_channels)
    ).tolist()

    ### ----- READOUTS -----

    # linear readout
    readout_1_weights = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(
        num_channels
    )
    output["readout_1_weights_for_M0"] = (
        weights_0[0] @ readout_1_weights * model.scale_shift.scale.item()
    ).tolist()

    # nonlinear readout
    # output["mlp_hidden_layers"] = 16 // TODO
    output["readout_2_weights_1"] = (
        torch.reshape(model.readouts[1].linear_1.weight, (num_channels, 16))
        .T.numpy(force=True)
        .flatten()
        / np.sqrt(num_channels)
    ).tolist()
    output["readout_2_weights_2"] = (
        model.readouts[1].linear_2.weight.numpy(force=True).flatten()
        / np.sqrt(16)
        * model.scale_shift.scale.item()
    ).tolist()
    act = model.readouts[1].non_linearity.acts[0]
    if act.f is not torch.nn.functional.silu:
        raise RuntimeError(
            "Currently, symmetrix only supports SiLU readout activations."
        )
    output["readout_2_scale_factor"] = act.cst

    return output


def symmetric_contraction(layer, module, elements, tolerance=1e-6):
    """Reformat MACE symmetric contraction into recursive form with equivariant
    intermediates, roughly in the style of NICE (Nigam, Pozdnyakov and Ceriotti,
    arXiv:2007.03407):

        B[l, m, k] = sum_{m1, m2} C(l1, m1, l2, m2; l, m) B'[l1, m1, k] A[l2, m2, k],

    with C representing coupling coefficients and k the channel. The output is

        M[L, M, k] = sum_eta w_eta[z, k] B_eta[L, M, k],

    with z the element. The couplings are chosen greedily, cheapest first, and
    MACE's weights are transformed exactly to this form using least squares.
    """
    irreps_in = [ir for _, ir in module.irreps_in]
    irreps_out = [ir for _, ir in module.irreps_out]
    offsets = np.cumsum([0] + [2 * ir.l + 1 for ir in irreps_in])
    correlation = module.contractions[0].correlation

    nodes = [
        dict(l=ir.l, p=ir.p, depth=1, j=j, parent=None, cg=[], poly=[])
        for j, ir in enumerate(irreps_in)
    ]
    for j, N in enumerate(nodes):
        N["poly"] = [{(offsets[j] + m,): 1.0} for m in range(2 * N["l"] + 1)]
    targets = {(ir.l, ir.p) for ir in irreps_out}
    for depth in range(2, correlation + 1):
        for N in [N for N in nodes if N["depth"] == depth - 1]:
            for j, ir in enumerate(irreps_in):
                if depth == 2 and j < N["j"]:
                    continue
                for l in range(abs(N["l"] - ir.l), N["l"] + ir.l + 1):
                    p = N["p"] * ir.p
                    if depth == correlation and (l, p) not in targets:
                        continue
                    w3j = wigner_3j(N["l"], ir.l, l, dtype=torch.float64).numpy()
                    cg = [
                        (M, m1, m2, w3j[m1, m2, M])
                        for m1, m2, M in zip(*np.nonzero(np.abs(w3j) > 1e-12))
                    ]
                    poly = [{} for _ in range(2 * l + 1)]
                    for M, m1, m2, c in cg:
                        for mono, v in N["poly"][m1].items():
                            key = tuple(sorted(mono + (offsets[j] + m2,)))
                            poly[M][key] = poly[M].get(key, 0.0) + c * v
                    if any(abs(v) > 1e-12 for P in poly for v in P.values()):
                        nodes.append(
                            dict(l=l, p=p, depth=depth, j=j, parent=N, cg=cg, poly=poly)
                        )

    used = set()

    def cost(N):
        if N["parent"] is None or id(N) in used:
            return 0
        return len(N["cg"]) + cost(N["parent"])

    basis = []
    for o, ir in enumerate(irreps_out):
        for depth in range(1, correlation + 1):
            U = module.contractions[o].U_tensors(depth).numpy(force=True)
            U = U.reshape(2 * ir.l + 1, *U.shape[-depth - 1 :])
            if not U.any():
                continue
            pool = [
                N
                for N in nodes
                if N["depth"] == depth and (N["l"], N["p"]) == (ir.l, ir.p)
            ]
            nonzero = list(zip(*np.nonzero(U)))
            monomials = {m for N in pool for P in N["poly"] for m in P}
            monomials |= {tuple(sorted(i[1:-1])) for i in nonzero}
            index = {m: i for i, m in enumerate(sorted(monomials))}

            def row(poly):
                r = np.zeros((2 * ir.l + 1) * len(index))
                for M, P in enumerate(poly):
                    for m, v in P.items():
                        r[M * len(index) + index[m]] += v
                return r

            Q, chosen = np.zeros((0, (2 * ir.l + 1) * len(index))), []
            while True:
                best = None
                for N in pool:
                    r = row(N["poly"])
                    r_new = r - Q.T @ (Q @ r)
                    independent = np.linalg.norm(r_new) > 1e-8 * np.linalg.norm(r)
                    if independent and (best is None or cost(N) < best[0]):
                        best = (cost(N), N, r_new)
                if best is None:
                    break
                Q = np.vstack([Q, best[2] / np.linalg.norm(best[2])])
                chosen.append(best[1])
                N = best[1]
                while N["parent"] is not None:
                    used.add(id(N))
                    N = N["parent"]
            V = np.zeros((U.shape[-1], (2 * ir.l + 1) * len(index)))
            for i in nonzero:
                V[i[-1], i[0] * len(index) + index[tuple(sorted(i[1:-1]))]] += U[i]
            B = np.array([row(N["poly"]) for N in chosen])
            R = np.linalg.lstsq(B.T, V.T, rcond=None)[0]
            if np.abs(B.T @ R - V.T).max() > tolerance * np.abs(V).max():
                raise RuntimeError(
                    f"M{layer}: the coupling trees do not span MACE's basis"
                )
            contraction = module.contractions[o]
            W = (
                contraction.weights_max
                if depth == correlation
                else contraction.weights[correlation - 1 - depth]
            ).numpy(force=True)[elements]
            for b, N in enumerate(chosen):
                basis.append((o, N, np.einsum("e,aek->ak", R[b], W)))

    values = {
        id(N): offsets[j] + np.arange(2 * N["l"] + 1)
        for j, N in enumerate(nodes[: len(irreps_in)])
    }
    num_values = offsets[-1]
    terms = {}
    for N in nodes:
        if id(N) not in used:
            continue
        values[id(N)] = num_values + np.arange(2 * N["l"] + 1)
        num_values += 2 * N["l"] + 1
        for M, m1, m2, c in N["cg"]:
            q, a, b = (
                values[id(N)][M],
                values[id(N["parent"])][m1],
                offsets[N["j"]] + m2,
            )
            key = (int(q), int(min(a, b)), int(max(a, b)))
            terms[key] = terms.get(key, 0.0) + float(c)
    out_offsets = np.cumsum([0] + [2 * ir.l + 1 for ir in irreps_out])
    logging.info(
        f"M{layer}: {len(basis)} basis polynomials, {num_values - offsets[-1]} values, "
        f"{len(terms)} terms"
    )
    return {
        f"B{layer}_spec": list(terms),
        f"B{layer}_coefficients": list(terms.values()),
        f"M{layer}_spec": [
            [int(out_offsets[o]), int(values[id(N)][0]), 2 * N["l"] + 1]
            for o, N, _ in basis
        ],
        f"M{layer}_weights": np.stack([w for _, _, w in basis], axis=1).tolist(),
    }


def first_layer_basis(
    model, atomic_numbers, output, num_spline_points, r_min, margin, bc_type
):
    mlp = model.interactions[0].conv_tp_weights
    zs = model.atomic_numbers.tolist()
    r, h = np.linspace(r_min, output["r_cut"], num_spline_points, retstep=True)
    x = (r[:-1, None] + h * np.array([0.25, 0.5, 0.75])).flatten()
    dx = 1e-5
    w = x[:, None] ** 2

    def rel(d, y):
        return np.sqrt(np.sum(w * d**2) / np.sum(w * y**2))

    def hidden(a, b, r):
        bessels = model.radial_embedding(
            torch.tensor(r, dtype=torch.float64).unsqueeze(-1),
            torch.eye(len(zs)),
            torch.tensor([[zs.index(a)], [zs.index(b)]]),
            model.atomic_numbers,
        )
        bessels, cutoff = bessels if isinstance(bessels, tuple) else (bessels, None)
        c = 1 if cutoff is None else cutoff
        out = []
        hook = list(mlp.children())[-1].register_forward_pre_hook(
            lambda m, i: out.append(i[0])
        )
        R = mlp(bessels)
        hook.remove()
        return (out[0] * c).numpy(force=True), 2 * np.sqrt(np.pi) * (R * c).numpy(
            force=True
        )

    T, S, n = len(atomic_numbers), len(r), len(x)
    pairs = [(i, j) for i in range(T) for j in range(i, T)]
    H, R = zip(
        *(
            hidden(
                atomic_numbers[i],
                atomic_numbers[j],
                np.concatenate([r, x, x + dx, x - dx]),
            )
            for i, j in pairs
        )
    )
    Hk, Rk = np.concatenate([y[:S] for y in H]), np.concatenate([y[:S] for y in R])
    U = np.linalg.lstsq(Hk, Rk, rcond=None)[0].T
    if np.linalg.norm(Hk @ U.T - Rk) > 1e-10 * np.linalg.norm(Rk):
        logging.warning(
            "No first-layer basis: the radial MLP's last layer is not linear"
        )
        return {}
    splines = [CubicSpline(r, y[:S], bc_type=bc_type) for y in H]
    C, N = output["num_channels"], U.shape[1]
    H0 = np.reshape(output["H0_weights"], (T, C))
    widths = [k for k in (32, 64) if k < N]
    basis, maps = {k: [] for k in widths}, {k: [] for k in widths}
    basis_error = {k: np.zeros(2) for k in widths}
    ratio = {k: np.zeros(2) for k in widths}
    spline_error = np.zeros(2)
    for a in range(T):
        for b in range(T):
            p = pairs.index((min(a, b), max(a, b)))
            Q = np.concatenate(
                [
                    (np.reshape(A0, (C, C)) * H0[b][:, None]).T @ U[l * C : (l + 1) * C]
                    for l, A0 in enumerate(output["A0_weights"][a])
                ]
            )
            Hv = H[p][S : S + n]
            Hd = (H[p][S + n : S + 2 * n] - H[p][S + 2 * n :]) / (2 * dx)
            Ov, Od = Hv @ Q.T, Hd @ Q.T
            se = [
                rel(splines[p](x) @ Q.T - Ov, Ov),
                rel(splines[p](x, 1) @ Q.T - Od, Od),
            ]
            G = (w * Hv).T @ Hv / (se[0] ** 2 * np.sum(w * Ov**2))
            G += (w * Hd).T @ Hd / (se[1] ** 2 * np.sum(w * Od**2))
            e, V = np.linalg.eigh(G)
            L = np.linalg.svd(
                Q @ (V * np.sqrt(np.maximum(e, 0))) @ V.T, full_matrices=False
            )[0]
            for k in widths:
                A = L[:, :k]
                B = A.T @ Q
                be = np.array(
                    [rel(Hv @ (A @ B - Q).T, Ov), rel(Hd @ (A @ B - Q).T, Od)]
                )
                basis_error[k] += be**2 / T**2
                ratio[k] = np.maximum(ratio[k], be / se)
                basis[k].append(B.flatten().tolist())
                maps[k].append(A.flatten().tolist())
            spline_error += np.square(se) / T**2
    spline_error = np.sqrt(spline_error)
    fits = [k for k in widths if margin is not None and np.all(ratio[k] <= margin)]
    width = fits[0] if fits else N
    logging.info(
        f"first-layer basis: {width} of {N} functions per pair; largest ratios of basis "
        f"to spline errors (values, derivatives): {ratio}"
    )
    data = {
        "first_layer_weights": U.flatten().tolist(),
        "first_layer_values": [y[:S].T.tolist() for y in H],
        "first_layer_derivs": [s(r, 1).T.tolist() for s in splines],
        "first_layer_basis_size": width,
        "first_layer_basis_error": {
            k: np.sqrt(v).tolist() for k, v in basis_error.items()
        },
        "first_layer_spline_error": spline_error.tolist(),
        "first_layer_basis_ratio": {k: v.tolist() for k, v in ratio.items()},
    }
    if width < N:
        data["first_layer_basis"], data["first_layer_map"] = basis[width], maps[width]
    return data
