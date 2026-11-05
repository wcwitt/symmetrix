import numpy as np
import torch
from scipy.interpolate import CubicSpline

from e3nn.o3 import Irreps


def _layout(irreps):
    """The e3nn indices of the layout [l][m][channel], with the channels of each l in
    irreps order, and the channels per l."""
    irreps = Irreps(irreps)
    mul = [sum(n for n, ir in irreps if ir.l == l) for l in range(irreps.lmax + 1)]
    off = np.cumsum([0] + [(2 * l + 1) * n for l, n in enumerate(mul)])
    seen = [0] * len(mul)
    perm = np.zeros(irreps.dim, int)
    e = 0
    for n, ir in irreps:
        for u in range(n):
            for m in range(2 * ir.l + 1):
                perm[off[ir.l] + m * mul[ir.l] + seen[ir.l] + u] = e
                e += 1
        seen[ir.l] += n
    return perm, mul, off


def _linear(f, irreps_in, irreps_out, scale=1.0):
    """An equivariant affine map in the layout [l][m][channel]: per output l, the
    flattened [channel in][channel out] weights, and the bias of the l = 0 outputs."""
    p, mul_in, off_in = _layout(irreps_in)
    q, mul_out, off_out = _layout(irreps_out)
    with torch.no_grad():
        b = f(torch.zeros(1, len(p), dtype=torch.float64))[0]
        Y = (f(torch.eye(len(p), dtype=torch.float64)) - b)[p][:, q].numpy()
    full = np.zeros_like(Y)
    W = []
    for l, n in enumerate(mul_out):
        k = mul_in[l] if l < len(mul_in) else 0
        w = np.zeros((0, n))
        if k:
            w = Y[off_in[l] : off_in[l] + k, off_out[l] : off_out[l] + n]
        for m in range(2 * l + 1 if k else 0):
            full[
                off_in[l] + m * k : off_in[l] + (m + 1) * k,
                off_out[l] + m * n : off_out[l] + (m + 1) * n,
            ] = w
        W.append((scale * w).flatten().tolist())
    b = b[q].numpy()
    assert np.abs(full - Y).max() <= 1e-12 * max(np.abs(Y).max(), 1)
    assert np.abs(b[mul_out[0] :]).max(initial=0) == 0
    return {
        "in": [mul_in[l] if l < len(mul_in) else 0 for l in range(len(mul_out))],
        "out": mul_out,
        "weights": W,
        "bias": (scale * b[: mul_out[0]]).tolist(),
    }


def _dot(f, irreps):
    """out[u] = sum_lm x[l][m][u] sum_v y[l][m][v] D_l[v][u], per l the flattened D_l"""
    p, mul, off = _layout(irreps)
    D = []
    for l, n in enumerate(mul):
        x = torch.zeros(n, len(p), dtype=torch.float64)
        x[:, p[off[l] : off[l] + n]] = 1.0
        y = torch.zeros(n, len(p), dtype=torch.float64)
        y[range(n), p[off[l] : off[l] + n]] = 1.0
        with torch.no_grad():
            D.append(f(x, y).numpy())
    x, y = torch.randn(2, 3, len(p), dtype=torch.float64)
    with torch.no_grad():
        ref = f(x, y).numpy()
    x = x[:, p].numpy()
    y = y[:, p].numpy()
    out = sum(
        (
            x[:, o : o + (2 * l + 1) * n].reshape(3, -1, n)
            * (y[:, o : o + (2 * l + 1) * n].reshape(3, -1, n) @ D[l])
        ).sum(1)
        for l, (n, o) in enumerate(zip(mul, off))
    )
    assert np.abs(out - ref).max() < 1e-10 * np.abs(ref).max()
    return [d.flatten().tolist() for d in D]


def _modulation(f, irreps):
    """out[l][m][u] = x[l][m][u] sum_v s[v] M_l[v][u], per l the flattened M_l"""
    p, mul, off = _layout(irreps)
    M = []
    for l, n in enumerate(mul):
        x = torch.zeros(mul[0], len(p), dtype=torch.float64)
        x[:, p[off[l] : off[l] + n]] = 1.0
        with torch.no_grad():
            y = f(x, torch.eye(mul[0], dtype=torch.float64))
        M.append(y[:, p][:, off[l] : off[l] + n].numpy())
    x = torch.randn(3, len(p), dtype=torch.float64)
    s = torch.randn(3, mul[0], dtype=torch.float64)
    with torch.no_grad():
        ref = f(x, s)[:, p].numpy()
    x = x[:, p].numpy()
    s = s.numpy()
    out = np.concatenate(
        [
            (
                x[:, o : o + (2 * l + 1) * n].reshape(3, -1, n) * (s @ M[l])[:, None]
            ).reshape(3, -1)
            for l, (n, o) in enumerate(zip(mul, off))
        ],
        1,
    )
    assert np.abs(out - ref).max() < 1e-10 * np.abs(ref).max()
    return [m.flatten().tolist() for m in M]


def _gate(g, irreps_in, num_scalars):
    """[l=0: cs silu(scalars), l>0: cg sigmoid(gates_l) x_l], the gates of the l > 0
    after the scalars"""
    p, mul, off = _layout(irreps_in)
    q, _, _ = _layout(g.irreps_out)
    x = torch.randn(3, len(p), dtype=torch.float64)
    with torch.no_grad():
        y = g(x)[:, q].numpy()
    x = x[:, p].numpy()
    silu = x[:, :num_scalars] / (1 + np.exp(-x[:, :num_scalars]))
    cs = y[0, 0] / silu[0, 0]
    sigmoid = 1 / (1 + np.exp(-x[:, num_scalars : mul[0]]))
    cg = y[0, num_scalars] / (sigmoid[0, 0] * x[0, mul[0]])
    out = [cs * silu]
    for l in range(1, len(mul)):
        s = sigmoid[:, sum(mul[1:l]) : sum(mul[1 : l + 1])]
        x_l = x[:, off[l] : off[l + 1]].reshape(3, -1, mul[l])
        out.append((cg * s[:, None, :] * x_l).reshape(3, -1))
    assert np.abs(np.concatenate(out, 1) - y).max() < 1e-12 * np.abs(y).max()
    return {"scalars": num_scalars, "csts": [float(cs), float(cg)]}


def _mlp(net):
    """Linear -> LayerNorm -> SiLU stacks, the last layer linear: per layer the flattened
    [in][out] weights, the bias, and the LayerNorm weight, bias and eps."""
    layers = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            layers.append(
                {
                    "weights": m.weight.T.flatten().tolist(),
                    "bias": m.bias.tolist(),
                    "in": m.in_features,
                    "out": m.out_features,
                }
            )
        elif isinstance(m, torch.nn.LayerNorm):
            layers[-1].update(
                ln_weight=m.weight.tolist(), ln_bias=m.bias.tolist(), ln_eps=m.eps
            )
        else:
            assert isinstance(m, torch.nn.SiLU)
    return layers


def _bias_readout(r, irreps_in, scale=1.0):
    """NonLinearBiasReadoutBlock: linear_1, act, linear_mid, act, linear_2"""
    x = torch.linspace(-4, 4, 9, dtype=torch.float64)[:, None].repeat(
        1, r.hidden_irreps.dim
    )
    with torch.no_grad():
        y = r.non_linearity(x)
    cst = (y[-1, 0] / torch.nn.functional.silu(x[-1, 0])).item()
    assert torch.allclose(y, cst * torch.nn.functional.silu(x), rtol=1e-12, atol=0)
    return {
        "linear_1": _linear(r.linear_1, irreps_in, r.hidden_irreps),
        "act": cst,
        "linear_mid": _linear(r.linear_mid, r.hidden_irreps, r.hidden_irreps),
        "linear_2": _linear(r.linear_2, r.hidden_irreps, r.linear_2.irreps_out, scale),
    }


def _gated_readout(r, irreps_in):
    """GeneralNonLinearBiasReadoutBlock: linear_1, gate, linear_mid, gate, linear_2"""
    num_scalars = r.hidden_irreps.count("0e")
    return {
        "linear_1": _linear(r.linear_1, irreps_in, r.irreps_nonlin),
        "gate": _gate(r.equivariant_nonlin, r.irreps_nonlin, num_scalars),
        "linear_mid": _linear(r.linear_mid, r.hidden_irreps, r.irreps_nonlin),
        "linear_2": _linear(r.linear_2, r.hidden_irreps, r.irreps_out),
    }


def _float64_U(model):
    """The products' U tensors in float64 where the model stores them rounded to float32"""
    from mace.tools.cg import U_matrix_real

    for product in model.products:
        s = product.symmetric_contractions
        irreps = Irreps([(1, ir) for _, ir in s.irreps_in])
        for o, c in enumerate(s.contractions):
            for nu in range(1, c.correlation + 1):
                U = c.U_tensors(nu)
                if torch.equal(U, U.float().to(U.dtype)):
                    ir = Irreps([(1, s.irreps_out[o].ir)])
                    V = U_matrix_real(irreps, ir, nu, dtype=torch.float64)[-1]
                    assert (V - U).abs().max() <= 1e-6 * U.abs().max()
                    setattr(c, f"U_matrix_{nu}", V)


def extract_mace_data_polar(model, atomic_numbers, num_spline_points, r_min):
    """PolarMACE: two RealAgnosticResidualNonLinearInteractionBlock layers with
    element-agnostic products, and the charge model (sources, Fukui functions, field
    updates, electron energy, k-space electrostatics). Features use the layout
    [l][m][channel] (see _layout)."""
    from graph_longrange.utils import FIELD_CONSTANT

    from .extract_mace_data import phi1_spec, symmetric_contraction

    bc_type = ("not-a-knot", "clamped")
    f64 = torch.float64
    zs = model.atomic_numbers.tolist()
    if len(atomic_numbers) == 0:
        atomic_numbers = zs
    atomic_numbers = sorted(atomic_numbers)
    T = len(atomic_numbers)
    attrs = torch.eye(len(zs), dtype=f64)[[zs.index(a) for a in atomic_numbers]]
    hidden = model.products[0].linear.irreps_out
    target = model.interactions[0].target_irreps
    charges = model.charges_irreps
    potential = model.potential_irreps
    C = hidden.count("0e")
    scale = torch.atleast_1d(model.scale_shift.scale)[0].item()
    shift = torch.atleast_1d(model.scale_shift.shift)[0].item()
    E0 = torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())
    blocks = [type(it).__name__ for it in model.interactions]
    assert blocks == ["RealAgnosticResidualNonLinearInteractionBlock"] * 2
    assert len(model.field_dependent_charges_maps) == 2
    assert model.spherical_harmonics._lmax == 3
    assert hidden == Irreps(f"{C}x0e+{C}x1o")
    assert all(
        it.target_irreps == Irreps(f"{C}x0e+{C}x1o+{C}x2e+{C}x3o")
        for it in model.interactions
    )
    assert all(p.use_agnostic_product for p in model.products)
    assert model.add_local_electron_energy and not model.field_si
    assert not model.quadrupole_feature_corrections
    assert not hasattr(model, "pair_repulsion")
    assert not model.electric_potential_descriptor.include_self_interaction
    assert model.coulomb_energy.include_self_interaction
    _float64_U(model)
    output = {
        "format_version": 3,
        "model_type": "PolarMACE",
        "num_channels": C,
        "r_cut": model.r_max.item(),
        "l_max": 3,
        "L_max": 1,
        "atomic_numbers": atomic_numbers,
        "num_elements": T,
        "atomic_energies": [E0[zs.index(a)].item() + shift for a in atomic_numbers],
    }

    ### ----- RADIAL FUNCTIONS AND DENSITIES -----

    # R = cutoff (W h + b), linear in phi = (cutoff h, cutoff) with h the last hidden
    # layer, by ordered pair; 2 sqrt(pi) of the spherical harmonics folded into phi
    r = np.linspace(r_min, output["r_cut"], num_spline_points)
    x = torch.tensor(r, dtype=f64)[:, None]
    bessel, cutoff = model.radial_embedding(x, None, None, model.atomic_numbers)
    output["radial_spline_r_min"] = float(r[0])
    alpha = [it.alpha.item() for it in model.interactions]
    for t, it in enumerate(model.interactions):
        last = it.conv_tp_weights.net[-1]
        weights = torch.cat([last.weight.T, last.bias[None]]).numpy(force=True)
        values = []
        derivs = []
        density_values = []
        density_derivs = []
        for a in range(T):
            for b in range(T):
                feats = torch.cat(
                    [
                        bessel,
                        it.source_embedding(attrs[[b] * len(r)]),
                        it.target_embedding(attrs[[a] * len(r)]),
                    ],
                    -1,
                )
                h = []
                hook = last.register_forward_pre_hook(lambda m, i: h.append(i[0]))
                with torch.no_grad():
                    R = (it.conv_tp_weights(feats) * cutoff).numpy()
                    d = (torch.tanh(it.density_fn(feats) ** 2) * cutoff).numpy()
                hook.remove()
                phi = torch.cat([h[0] * cutoff, cutoff], -1).numpy(force=True)
                assert np.abs(phi @ weights - R).max() < 1e-10 * np.abs(R).max()
                spl = CubicSpline(r, 2 * np.sqrt(np.pi) * phi, axis=0, bc_type=bc_type)
                values.append(spl(r).T.tolist())
                derivs.append(spl(r, 1).T.tolist())
                spl = CubicSpline(
                    r, d[:, 0] * it.beta.item() / alpha[t], bc_type=bc_type
                )
                density_values.append(spl(r).tolist())
                density_derivs.append(spl(r, 1).tolist())
        output[f"radial_spline_values_{t}"] = values
        output[f"radial_spline_derivs_{t}"] = derivs
        output[f"A{t}_spline_values"] = density_values
        output[f"A{t}_spline_derivs"] = density_derivs
        output[f"gate_{t}"] = _gate(it.equivariant_nonlin, it.irreps_nonlin, C)
        output[f"linear_2_{t}"] = _linear(it.linear_2, target, target)
        if t == 1:
            output["radial_weights_1"] = weights.tolist()

    ### ----- FIRST LAYER -----

    # X0 = sum_j phi(r_ij) Y(ij) X0_weights[z_j] / s + residual[z_i], s = 1 + sum density,
    # with X0_weights[b][n][l-block of X0]
    it = model.interactions[0]
    with torch.no_grad():
        h0 = model.node_embedding(attrs)
        x = it.linear_up(h0)
        residual = it.linear_res(x).numpy()
        skip = it.skip_tp(h0).numpy()
        Y = torch.zeros(1, 16, dtype=f64)
        c = []
        for l in range(4):
            Y.zero_()
            Y[0, l * l] = 1.0
            w = torch.ones(1, 4 * C, dtype=f64)
            c.append(it.conv_tp(torch.ones(1, C, dtype=f64), Y, w)[0, l * l * C].item())
        Y = torch.randn(1, 16, dtype=f64)
        w = torch.randn(1, 4 * C, dtype=f64)
        xr = torch.randn(1, C, dtype=f64)
        ref = it.conv_tp(xr, Y, w)[0].numpy()
    p, _, _ = _layout(it.conv_tp.irreps_out)
    mine = np.concatenate(
        [
            (
                c[l]
                * w[0, l * C : (l + 1) * C, None]
                * xr[0, :, None]
                * Y[0, None, l * l : (l + 1) ** 2]
            )
            .numpy()
            .T.flatten()
            for l in range(4)
        ]
    )
    assert np.abs(mine - ref[p]).max() < 1e-12 * np.abs(ref).max()
    nonlin = _layout(it.irreps_nonlin)[1]
    assert (
        np.abs(residual[:, nonlin[0] :]).max() == 0 and np.abs(skip[:, C:]).max() == 0
    )
    linear_1 = _linear(it.linear_1, it.conv_tp.irreps_out, it.irreps_nonlin)
    assert not any(linear_1["bias"])
    W1 = linear_1["weights"]
    last = it.conv_tp_weights.net[-1]
    weights = torch.cat([last.weight.T, last.bias[None]]).numpy(force=True)
    output["X0_weights"] = [
        np.concatenate(
            [
                (weights[:, l * C : (l + 1) * C] * (c[l] * x[b].numpy() / alpha[0]))
                @ np.reshape(W1[l], (C, nonlin[l]))
                for l in range(4)
            ],
            1,
        ).tolist()
        for b in range(T)
    ]
    output["residual_0"] = residual[:, : nonlin[0]].tolist()
    output["skip_0"] = skip[:, :C].tolist()

    ### ----- SECOND LAYER -----

    it = model.interactions[1]
    C1 = it.edge_irreps.count("0e")
    assert it.edge_irreps == Irreps(f"{C1}x0e+{C1}x1o")
    output["num_edge_channels"] = C1
    output["linear_up_1"] = _linear(it.linear_up, hidden, it.edge_irreps)
    output["residual_1"] = _linear(it.linear_res, it.edge_irreps, it.irreps_nonlin)
    output["skip_1"] = _linear(it.skip_tp, hidden, hidden)
    output.update(phi1_spec(it.conv_tp, C1, 3, 1))
    output["linear_1_1"] = _linear(
        it.linear_1, it.conv_tp.irreps_out, it.irreps_nonlin, 1 / alpha[1]
    )

    ### ----- PRODUCTS AND READOUTS -----

    for t, product in enumerate(model.products):
        output.update(symmetric_contraction(t, product.symmetric_contractions, [0]))
        output[f"product_linear_{t}"] = _linear(product.linear, hidden, hidden)
    readout = _linear(model.readouts[0].linear, hidden, "0e", scale)
    assert not any(readout["bias"])
    output["readout_0"] = readout["weights"][0]
    output["readout_1"] = _bias_readout(model.readouts[1], hidden, scale)

    ### ----- CHARGES -----

    assert not any(s.zero_charges for s in model.lr_source_maps)
    output["sources"] = [
        _linear(s.linear, hidden, charges) for s in model.lr_source_maps
    ]
    output["mixer"] = [
        _linear(m, hidden, hidden) for m in model.layer_feature_mixer.linears
    ]
    output["fukui"] = _bias_readout(model.fukui_source_map, hidden)
    output["updates"] = []
    for u in model.field_dependent_charges_maps:
        assert type(u).__name__ == "AgnosticEmbeddedOneBodyVariableUpdate"
        pe = u.potential_embedding
        with torch.no_grad():
            source = u.source_embedding(attrs).tolist()
        output["updates"].append(
            {
                "potential": _linear(pe.potential_linear, potential, hidden),
                "node": _linear(pe.node_feats_linear, hidden, hidden),
                "charge": _linear(pe.charge_embedding, charges, hidden),
                "source": source,
                "dot": _dot(u.dot_products, hidden),
                "mlp": _mlp(u.nonlinearity.net),
                "out": _modulation(u.tp_out, hidden),
                "readout": _gated_readout(u.readout, hidden),
            }
        )
    e = model.local_electron_energy
    assert type(e).__name__ == "OneBodyMLPFieldReadout"
    output["electron"] = {
        "q": _linear(e.linear_up_q, charges, hidden),
        "v": _linear(e.linear_up_v, potential, hidden),
        "dot_q": _dot(e.dot_products_q, hidden),
        "dot_v": _dot(e.dot_products_v, hidden),
        "mlp": _mlp(e.mlp.net),
    }

    ### ----- ELECTROSTATICS -----

    # GTO Fourier coefficients a_l f_l(k) Y_lm(k) (1, -i)_l, f_0 = exp(-sigma^2 k^2/2),
    # f_1 = k f_0
    def basis(g):
        rs = g.radial_spline
        a = (torch.stack([rs.pref0, rs.pref1], -1) * g.cl_scale).tolist()
        assert g.real_phase_factors.tolist() == [1, 0, 0, 0]
        assert g.imag_phase_factors.tolist() == [0, -1, -1, -1]
        return {"sigmas": rs.sigma2.sqrt().tolist(), "a": a}

    fd = model.electric_potential_descriptor
    assert fd.density_basis.max_l == 1 and fd.feature_basis.max_l == 1
    output["kspace_cutoff"] = model.kspace_cutoff.item()
    output["field_constant"] = FIELD_CONSTANT
    output["density_basis"] = basis(fd.density_basis)
    output["feature_basis"] = basis(fd.feature_basis)
    assert basis(model.coulomb_energy.density_basis) == output["density_basis"]
    # features [lm][sigma], from [l][sigma][m]
    S = len(output["feature_basis"]["sigmas"])
    perm = [
        [l * l * S + s * (2 * l + 1) + lm - l * l for s in range(S)]
        for l in range(2)
        for lm in range(l * l, (l + 1) ** 2)
    ]
    select = fd.self_interaction_terms.select_indices.tolist()
    assert select == [
        lm for l in range(2) for s in range(S) for lm in range(l * l, (l + 1) ** 2)
    ]
    overlap = fd.self_interaction_terms.overlap_constants.numpy(force=True)
    output["field_self"] = overlap[perm].tolist()
    output["field_norms"] = model.field_feature_norms.numpy(force=True)[perm].tolist()
    return output
