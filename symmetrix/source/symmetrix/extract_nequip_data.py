import copy
import logging

import numpy as np
import torch

from ase.data import chemical_symbols


def extract_nequip_data(model, species, num_spline_points=256):
    """Extract data from a NequIP model into structure that can be
    written as symmetrix JSON data file

    Parameters
    ----------
    model: str / Path / torch.nn.Module
        NequIP package (.nequip.zip), checkpoint (.ckpt), nequip.net model ID
        (nequip.net:group-name/model-name:version), or a model in memory (not modified)
    species: list(int / str)
        list of atomic numbers or chemical symbols to extract (all of the model's types if empty)
    num_spline_points: int, default 256
        number of spline points of the radial functions

    Returns
    -------
    output_data: dict with symmetrix model data
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return _extract_nequip_data(model, species, num_spline_points)
    finally:
        torch.set_default_dtype(default_dtype)


def _extract_nequip_data(model, species, num_spline_points):
    from e3nn import o3
    from nequip.model.saved_models.load_utils import load_saved_model

    if isinstance(model, torch.nn.Module):
        model = copy.deepcopy(model)
    else:
        model = load_saved_model(
            str(model), compile_mode="eager", model_key="sole_model"
        )
    f = model.double().eval().model.func

    ### ----- CHECK FOR COMPATIBILITY -----

    convnets = []
    while hasattr(f, f"layer{len(convnets)}_convnet"):
        convnets.append(getattr(f, f"layer{len(convnets)}_convnet"))
    if len(convnets) == 0:
        raise RuntimeError("Not a NequIP model: no layer<i>_convnet found.")
    for c in convnets:
        if c.resnet or not hasattr(c.equivariant_nonlin, "act_scalars"):
            raise RuntimeError(
                "Currently, symmetrix only supports gated ConvNetLayers without resnet."
            )
    pair_potential = getattr(f, "pair_potential", None)
    if pair_potential is not None and type(pair_potential).__name__ != "ZBL":
        raise RuntimeError(
            "Currently, symmetrix only supports the ZBL pair potential, "
            f"found {type(pair_potential).__name__}."
        )
    if getattr(f.edge_norm, "_per_edge_type", False):
        raise RuntimeError(
            "Currently, symmetrix does not support per-edge-type cutoffs."
        )

    def scatter_norm(conv):
        if hasattr(conv, "scatter_norm_factor"):
            return 1.0 if conv.scatter_norm_factor is None else conv.scatter_norm_factor
        n = conv.avg_num_neighbors_norm.norm_const
        if n.numel() != 1:
            raise RuntimeError(
                "Currently, symmetrix only supports one avg_num_neighbors for all types."
            )
        return float(n)

    alphas = {scatter_norm(c.conv) for c in convnets}
    alpha = float(next(iter(alphas)))
    if len(alphas) != 1:
        raise RuntimeError(
            "Currently, symmetrix only supports one scatter normalization "
            f"for all layers, found {alphas}."
        )
    even = set()
    odd = set()
    for c in convnets:
        gate = c.equivariant_nonlin
        acts = list(zip(gate.irreps_scalars, gate.act_scalars.acts))
        if gate.act_gates is not None:
            if any(irr.p != 1 for _, irr in gate.irreps_gates):
                raise RuntimeError("Currently, symmetrix only supports even gates.")
            acts += list(zip(gate.irreps_gates, gate.act_gates.acts))
        order = [2 * irr.l + (irr.p == -1) for _, irr in gate.irreps_gated]
        if order != sorted(set(order)):
            raise RuntimeError(
                "Currently, symmetrix only supports gated irreps in the order "
                f"1e, 1o, 2e, 2o, ..., found {gate.irreps_gated}."
            )
        for (_, irr), a in acts:
            (even if irr.p == 1 else odd).add((getattr(a.f, "__name__", ""), a.cst))
    if (
        len(even) != 1
        or len(odd) > 1
        or next(iter(even))[0] != "silu"
        or any(n != "tanh" for n, _ in odd)
    ):
        raise RuntimeError(
            "Currently, symmetrix only supports SiLU (even) and tanh (odd) activations "
            f"with one normalization constant each, found {even} {odd}."
        )

    ### ----- BASIC MODEL INFO -----

    names = list(f.per_type_energy_scale_shift.type_names)
    has_zbl = pair_potential is not None
    r_cut = float(f.edge_norm.r_max)
    output = {}
    output["model_type"] = "NequIP"
    output["num_layers"] = len(convnets)
    output["r_cut"] = r_cut
    output["gate_cst"] = float(next(iter(even))[1])
    if odd:
        output["gate_cst_odd"] = float(next(iter(odd))[1])

    ### ----- ATOMIC NUMBERS -----

    def atomic_number(name):
        if has_zbl:
            return int(round(float(pair_potential.atomic_numbers[names.index(name)])))
        return chemical_symbols.index(name)

    if species is None:
        species = []
    symbols = []
    for sp in species:
        try:
            symbols.append(chemical_symbols[int(sp)])
        except ValueError:
            symbols.append(str(sp))
    if len(symbols) == 0:
        symbols = list(names)
        logging.warning(f"No species, including all: {symbols}")
    for s in symbols:
        if s not in names:
            raise ValueError(f"{s} is not one of the model's types {names}")
    symbols = sorted(symbols, key=atomic_number)
    types = [names.index(s) for s in symbols]
    output["atomic_numbers"] = [atomic_number(s) for s in symbols]
    output["num_elements"] = len(symbols)

    ### ----- ZBL -----

    output["has_zbl"] = has_zbl
    if has_zbl:
        cutoff = getattr(pair_potential, "cutoff", f.bessel_encode.cutoff)
        output["zbl_prefactor"] = float(pair_potential._qqr2exesquare)
        output["zbl_p"] = float(cutoff.p)

    ### ----- RADIAL SPLINES -----

    r = torch.linspace(0, r_cut, num_spline_points).clamp(min=1e-6).requires_grad_(True)
    x = (r / r_cut).view(-1, 1)
    bw = f.bessel_encode.bessel_weights
    basis = (torch.sinc(x * bw) * bw) * f.bessel_encode.cutoff(x) * f.factor.factor
    spline_values = []
    spline_derivatives = []
    for c in convnets:
        R = c.conv.edge_mlp(basis)
        D = torch.zeros_like(R)
        for w in range(R.shape[1]):
            (D[:, w],) = torch.autograd.grad(R[:, w].sum(), r, retain_graph=True)
        spline_values.append(R.detach().numpy().T)
        spline_derivatives.append(D.detach().numpy().T)
    output["radial_spline_values"] = [
        (2 * np.sqrt(np.pi) * R).tolist() for R in spline_values
    ]
    output["radial_spline_derivs"] = [
        (2 * np.sqrt(np.pi) * D).tolist() for D in spline_derivatives
    ]

    ### ----- EMBEDDING -----

    E = f.type_embed.embed_module.weight.detach()[types]
    output["embedding"] = E.numpy().tolist()

    ### ----- LAYERS -----

    def irreps_list(ir):
        return [[mul, irr.l] for mul, irr in ir]

    def irreps_parity(ir):
        return [irr.p for _, irr in ir]

    two_parities = any(
        len({irr.p for _, irr in ir if irr.l == l}) > 1
        for c in convnets
        for ir in (
            c.conv.linear_1.irreps_in,
            c.conv.linear_2.irreps_out,
            c.conv.tp_scatter.tp.irreps_out,
        )
        for l in range(ir.lmax + 1)
    )

    def irrep_name(irr):
        return f"{irr.l}{'e' if irr.p == 1 else 'o'}" if two_parities else str(irr.l)

    def merged(ir):
        order = []
        comps = {}
        o = 0
        for mul, irr in ir:
            if irr not in comps:
                order.append(irr)
                comps[irr] = []
            comps[irr].append(np.arange(o, o + mul * irr.dim))
            o += mul * irr.dim
        perm = np.concatenate([np.concatenate(comps[irr]) for irr in order])
        return o3.Irreps(
            [(sum(len(c) for c in comps[irr]) // irr.dim, irr) for irr in order]
        ), perm

    def probe(linear, dim, *args):
        with torch.no_grad():
            return linear(torch.eye(dim), *args).numpy()

    def blocks(W, ir_in, ir_out):
        offsets_in = {}
        o = 0
        for mul, irr in ir_in:
            offsets_in.setdefault(irr, []).append((o, mul))
            o += mul * irr.dim
        offsets_out = {}
        o = 0
        for mul, irr in ir_out:
            offsets_out.setdefault(irr, []).append((o, mul))
            o += mul * irr.dim
        result = {}
        for irr in offsets_out:
            if irr not in offsets_in:
                continue
            if len(offsets_in[irr]) > 1 or len(offsets_out[irr]) > 1:
                raise RuntimeError(
                    "Currently, symmetrix only supports one block per irrep."
                )
            ((a, mul_in),) = offsets_in[irr]
            ((b, mul_out),) = offsets_out[irr]
            W_irr = W[a : a + mul_in * irr.dim, b : b + mul_out * irr.dim].reshape(
                mul_in, irr.dim, mul_out, irr.dim
            )
            for m in range(irr.dim):
                if not np.allclose(W_irr[:, m, :, m], W_irr[:, 0, :, 0]):
                    raise RuntimeError("Linear not uniform in m.")
            result[irrep_name(irr)] = W_irr[:, 0, :, 0].tolist()
        return result

    layers = []
    for c in convnets:
        conv = c.conv
        ir_in = conv.linear_1.irreps_in.simplify()
        ir_out, perm = merged(conv.linear_2.irreps_out.simplify())
        tp = conv.tp_scatter.tp
        if list(tp.irreps_in1) != list(ir_in):
            raise RuntimeError("Tensor-product input does not match linear_1.")
        layer = {}
        layer["irreps_in"] = irreps_list(ir_in)
        layer["irreps_out"] = irreps_list(ir_out)
        gate = c.equivariant_nonlin
        layer["gate"] = [
            gate.irreps_scalars.num_irreps,
            gate.irreps_gates.num_irreps if gate.irreps_gates is not None else 0,
        ]
        layer["linear_1"] = blocks(
            probe(conv.linear_1, ir_in.dim),
            ir_in,
            conv.linear_1.irreps_out.simplify(),
        )
        layer["linear_2"] = blocks(
            probe(conv.linear_2, conv.linear_2.irreps_in.dim)[:, perm],
            conv.linear_2.irreps_in.simplify(),
            ir_out,
        )
        layer["self_connection"] = {}
        if conv.sc is not None:
            for attrs in E:
                W = probe(conv.sc, ir_in.dim, attrs.view(1, -1).expand(ir_in.dim, -1))
                for k, W_k in blocks(W[:, perm], ir_in, ir_out).items():
                    layer["self_connection"].setdefault(k, []).append(W_k)
        ir1 = irreps_list(tp.irreps_in1)
        ir2 = irreps_list(tp.irreps_in2)
        irm = irreps_list(tp.irreps_out)
        o1 = np.cumsum([0] + [mul * (2 * l + 1) for mul, l in ir1])
        o2 = np.cumsum([0] + [mul * (2 * l + 1) for mul, l in ir2])
        om = np.cumsum([0] + [mul * (2 * l + 1) for mul, l in irm])
        layer["tp_instructions"] = []
        layer["tp_irreps_mid"] = irm
        layer["tp_coefficients"] = []
        w = 0
        for ins in tp.instructions:
            mul_1, l1 = ir1[ins.i_in1]
            l2 = ir2[ins.i_in2][1]
            mul_3, l3 = irm[ins.i_out]
            if ins.connection_mode != "uvu" or mul_1 != mul_3:
                raise RuntimeError(
                    "Currently, symmetrix only supports uvu tensor-product instructions."
                )
            C = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1))
            for m1 in range(2 * l1 + 1):
                for m2 in range(2 * l2 + 1):
                    x1 = torch.zeros(1, tp.irreps_in1.dim)
                    x1[0, o1[ins.i_in1] + m1] = 1
                    x2 = torch.zeros(1, tp.irreps_in2.dim)
                    x2[0, o2[ins.i_in2] + m2] = 1
                    weight = torch.zeros(1, tp.weight_numel)
                    weight[0, w] = 1
                    with torch.no_grad():
                        y = tp(x1, x2, weight)[0]
                    C[m1, m2] = y[om[ins.i_out] : om[ins.i_out] + 2 * l3 + 1].numpy()
            layer["tp_instructions"].append(
                [ins.i_in1, ins.i_in2, ins.i_out, l1, l2, l3, mul_1, w]
            )
            layer["tp_coefficients"].append(C.tolist())
            w += mul_1
        if w != tp.weight_numel:
            raise RuntimeError("Tensor-product weights do not match the instructions.")
        if two_parities:
            layer["irreps_in_parity"] = irreps_parity(ir_in)
            layer["irreps_out_parity"] = irreps_parity(ir_out)
            layer["tp_irreps_mid_parity"] = irreps_parity(tp.irreps_out)
        layers.append(layer)

    ### ----- READOUT -----

    num_readout = convnets[-1].conv.linear_2.irreps_out.dim
    mlp = f.per_atom_energy_readout.mlp_module
    with torch.no_grad():
        u = torch.randn(2, num_readout, generator=torch.Generator().manual_seed(0))
        if not torch.allclose(mlp(u[:1] + u[1:]), mlp(u[:1]) + mlp(u[1:])):
            raise RuntimeError(
                "Currently, symmetrix only supports a linear readout "
                "(readout_mlp_hidden_layers_depth=0)."
            )
        output["readout_weights"] = mlp(torch.eye(num_readout))[:, 0].tolist()
    scale_shift = f.per_type_energy_scale_shift
    scales = scale_shift.scales if scale_shift.has_scales else torch.ones(1)
    shifts = scale_shift.shifts if scale_shift.has_shifts else torch.zeros(1)
    output["per_type_scale"] = (
        scales.detach().reshape(-1).expand(len(names))[types].tolist()
    )
    output["per_type_shift"] = (
        shifts.detach().reshape(-1).expand(len(names))[types].tolist()
    )

    ### ----- IRREPS: 2l (EVEN) AND 2l+1 (ODD) -----

    num_layers = len(layers)
    num_types = len(types)
    l_max = max(max(row[3:6]) for layer in layers for row in layer["tp_instructions"])

    def irrep(layer, key, b):
        return 2 * layer[key][b][1] + (two_parities and layer[key + "_parity"][b] == -1)

    def multiplicities(layer, key, merge=False):
        mul = [0] * (2 * l_max + 2)
        for b, (m, _) in enumerate(layer[key]):
            if mul[irrep(layer, key, b)] > 0 and not merge:
                raise RuntimeError(
                    "Currently, symmetrix only supports one block per irrep."
                )
            mul[irrep(layer, key, b)] += m
        return mul

    def offsets(mul):
        sizes = [(2 * (i // 2) + 1) * m for i, m in enumerate(mul)]
        return np.cumsum([0] + sizes).tolist()

    H_mul = [multiplicities(layer, "irreps_in") for layer in layers]
    A_mul = [multiplicities(layer, "irreps_out") for layer in layers]
    Phi_mul = [multiplicities(layer, "tp_irreps_mid", merge=True) for layer in layers]
    for t, layer in enumerate(layers):
        num_scalars, num_gates = layer["gate"]
        gated = [num_scalars - A_mul[t][1]] + A_mul[t][1:]
        if sum(A_mul[t][2:]) != num_gates or gated[0] + num_gates != A_mul[t][0]:
            raise RuntimeError("Gates do not match the gated irreps.")
        if t < num_layers - 1 and gated != H_mul[t + 1]:
            raise RuntimeError("The irreps of consecutive layers do not match.")
    H_mul.append(gated)
    H_offset = [offsets(mul) for mul in H_mul]
    A_offset = [offsets(mul) for mul in A_mul]
    Phi_offset = [offsets(mul) for mul in Phi_mul]
    if (
        np.shape(output["embedding"]) != (num_types, H_offset[0][-1])
        or H_mul[0][0] != H_offset[0][-1]
    ):
        raise RuntimeError("The embedding does not match the first layer.")
    if len(output["readout_weights"]) != H_offset[-1][-1]:
        raise RuntimeError("The readout does not match the last layer.")
    output["l_max"] = l_max
    output["H_mul"] = H_mul
    output["H_offset"] = H_offset
    output["A_mul"] = A_mul
    output["A_offset"] = A_offset
    output["Phi_mul"] = Phi_mul
    output["Phi_offset"] = Phi_offset

    ### ----- LINEAR WEIGHTS PER IRREP -----

    def weights(layer, key, scale=1.0):
        w = [[] for _ in range(2 * l_max + 2)]
        for i in range(2 * l_max + 2):
            name = f"{i // 2}{'eo'[i % 2]}" if two_parities else str(i // 2)
            if (two_parities or i % 2 == 0) and name in layer[key]:
                w[i] = (scale * np.asarray(layer[key][name])).flatten().tolist()
        return w

    output["X_weights"] = [weights(layer, "linear_1", alpha) for layer in layers]
    output["A_weights"] = [weights(layer, "linear_2") for layer in layers]
    output["A_sc_weights"] = [weights(layer, "self_connection") for layer in layers]

    ### ----- TENSOR PRODUCT -----

    def by_column(spec, cg, column, width):
        order = sorted(range(len(spec)), key=lambda p: spec[p][column])
        spec = [spec[p] for p in order]
        cg = [cg[p] for p in order]
        rows = [
            p
            for p in range(len(spec))
            if p == 0 or spec[p][column] != spec[p - 1][column]
        ]
        columns = [-1] * width
        for q, p in enumerate(rows):
            for k in range(spec[p][4]):
                columns[spec[p][column] + k] = q
        return spec, cg, rows + [len(spec)], columns

    keys = ["Phi_spec", "Phi_clebsch_gordan", "Phi_rows", "Phi_columns"]
    for key in keys + [key + "_trans" for key in keys]:
        output[key] = []
    for t, layer in enumerate(layers):
        spec = []
        cg = []
        for q, row in enumerate(layer["tp_instructions"]):
            i_1 = irrep(layer, "irreps_in", row[0])
            i_3 = irrep(layer, "tp_irreps_mid", row[2])
            column = sum(
                m
                for o, (m, _) in enumerate(layer["tp_irreps_mid"][: row[2]])
                if irrep(layer, "tp_irreps_mid", o) == i_3
            )
            C = np.asarray(layer["tp_coefficients"][q])
            for m1, m2, m3 in zip(*np.nonzero(C)):
                spec.append(
                    [
                        H_offset[t][i_1] + int(m1) * H_mul[t][i_1],
                        row[4] * row[4] + int(m2),
                        row[7],
                        Phi_offset[t][i_3] + int(m3) * Phi_mul[t][i_3] + column,
                        row[6],
                    ]
                )
                cg.append(float(C[m1, m2, m3]))
        for suffix, values in [
            ("", by_column(spec, cg, 3, Phi_offset[t][-1])),
            ("_trans", by_column(spec, cg, 0, H_offset[t][-1])),
        ]:
            for key, value in zip(keys, values):
                output[key + suffix].append(value)

    return output
