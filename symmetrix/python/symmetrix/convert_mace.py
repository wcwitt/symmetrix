import torch
torch.serialization.add_safe_globals([slice])

from e3nn.o3 import Irrep, Irreps, Linear, wigner_3j
import itertools
from mace.modules.radial import ZBLBasis
from mace.tools.cg import U_matrix_real
from mace.tools.scripts_utils import remove_pt_head
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import CubicSpline
import logging

def extract_model_data(model_file, atomic_numbers, head=None, num_spline_points=200):
    model = torch.load(
        model_file,
        map_location=torch.device('cpu'),
        weights_only=False
    ).to(torch.float64)

    ### ----- EXTRACT SINGLE HEAD -----

    if hasattr(model, 'heads') and len(model.heads) != 1:
        torch.set_default_dtype(next(model.parameters()).dtype)
        model = remove_pt_head(model, head)

    ### ----- CHECK FOR COMPATIBILITY -----

    if len(model.interactions) != 2:
        raise RuntimeError("Currently, symmetrix only supports two-layer MACE models.")

    from mace.modules.blocks import RealAgnosticInteractionBlock, RealAgnosticDensityInteractionBlock
    if (not isinstance(model.interactions[0], RealAgnosticInteractionBlock)
        and
        not isinstance(model.interactions[0], RealAgnosticDensityInteractionBlock)):
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose first interaction is "
            "RealAgnosticInteractionBlock or RealAgnosticDensityInteractionBlock.")

    from mace.modules.blocks import RealAgnosticResidualInteractionBlock, RealAgnosticDensityResidualInteractionBlock
    if (not isinstance(model.interactions[1], RealAgnosticResidualInteractionBlock)
        and
        not isinstance(model.interactions[1], RealAgnosticDensityResidualInteractionBlock)):
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose second interaction is "
            "RealAgnosticResidualInteractionBlock or RealAgnosticDensityResidualInteractionBlock.")

    if (model.spherical_harmonics._lmax != 3):
        raise RuntimeError("Currently, symmetrix only supports MACE models with l_max=3.")

    ### ----- HELPER FUNCTION -----

    def linear_simplify(linear):
        simplified = Linear(Irreps(linear.irreps_in).simplify(),
                            Irreps(linear.irreps_out).simplify())
        simplified.weight = linear.weight
        simplified.bias = linear.bias
        return simplified

    ### ----- BASIC MODEL INFO -----

    num_channels = model.node_embedding.linear.irreps_out.count("0e")
    r_cut = model.r_max.item()
    l_max = model.spherical_harmonics._lmax
    L_max =  model.products[0].linear.irreps_out.lmax
    output = {}
    output['num_channels'] = num_channels
    output['r_cut'] = r_cut
    output['l_max'] = l_max
    output['L_max'] = L_max

    ### ----- ATOMIC NUMBERS AND ENERGIES -----

    atomic_numbers = sorted(atomic_numbers)
    if len(atomic_numbers) == 0:
        atomic_numbers = sorted(model.atomic_numbers.tolist())
        logging.warning(f"No atomic_numbers, including all: {atomic_numbers}")
    atomic_energies = [
        torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())[model.atomic_numbers.tolist().index(a)].item()
            + model.scale_shift.shift.item()
        for a in atomic_numbers]
    output['atomic_numbers'] = atomic_numbers
    output['num_elements'] = len(atomic_numbers)
    output['atomic_energies'] = atomic_energies

    ### --- ZBL ---

    if hasattr(model, "pair_repulsion") and model.pair_repulsion:
        if not isinstance(model.pair_repulsion_fn, ZBLBasis):
            raise Exception("Only ZBL pair_repulsion is supported.")
        output['has_zbl'] = True
        zbl = model.pair_repulsion_fn
        output['zbl_a_exp'] = zbl.a_exp.item()
        output['zbl_a_prefactor'] = zbl.a_prefactor.item()
        output['zbl_c'] = (model.scale_shift.scale.item() * zbl.c.numpy(force=True)).tolist()
        output['zbl_covalent_radii'] = zbl.covalent_radii.numpy(force=True).tolist()
        output['zbl_p'] = zbl.p.item()
    else:
        output['has_zbl'] = False

    ### ----- RADIAL SPLINES -----

    logging.info("R0+R1")
    # TODO: what to do about the 0.5 Ang buffer?
    r,h = np.linspace(1e-12, r_cut+0.5, num_spline_points, retstep=True)
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
                torch.tensor(r, dtype=torch.get_default_dtype()).unsqueeze(-1),
                torch.eye(len(model.atomic_numbers)),
                torch.tensor([[model_i],[model_j]], dtype=torch.int64),
                model.atomic_numbers)
            if isinstance(bessels, tuple):
                bessels = bessels[0]  # newer versions return (bessels, cutoffs)
            # radial basis for interaction 0
            R = model.interactions[0].conv_tp_weights(bessels).numpy(force=True)
            spl_0 = [CubicSpline(r, R[:,k]) for k in range(R.shape[1])]
            spline_values_0.append([spl(r).tolist() for spl in spl_0])
            spline_derivatives_0.append([spl.derivative()(r).tolist() for spl in spl_0])
            # radial basis for interaction 1
            R = model.interactions[1].conv_tp_weights(bessels).numpy(force=True)
            spl_1 = [CubicSpline(r, R[:,k]) for k in range(R.shape[1])]
            spline_values_1.append([spl(r).tolist() for spl in spl_1])
            spline_derivatives_1.append([spl.derivative()(r).tolist() for spl in spl_1])
    output['radial_spline_h'] = float(h)
    output['radial_spline_values_0'] = spline_values_0
    output['radial_spline_derivs_0'] = spline_derivatives_0
    output['radial_spline_values_1'] = spline_values_1
    output['radial_spline_derivs_1'] = spline_derivatives_1

    ### ----- H0 -----

    logging.info("H0")
    H0_weights = (
        np.reshape(model.node_embedding.linear.weight.numpy(force=True),
                   [len(model.atomic_numbers),num_channels]) / np.sqrt(len(model.atomic_numbers))
        @
        np.reshape(model.interactions[0].linear_up.weight.numpy(force=True),
                   [num_channels,num_channels]) / np.sqrt(num_channels)
        )
    indices = [model.atomic_numbers.tolist().index(a) for a in atomic_numbers]
    H0_weights = H0_weights[indices,:]
    output['H0_weights'] = H0_weights.flatten().tolist()

    ### ----- Phi0 -----

    logging.info("Phi0")

    ### ----- A0 -----

    logging.info("A0")
    A0_scaled = True if ("Density" in type(model.interactions[0]).__name__) else False
    output['A0_scaled'] = A0_scaled
    if A0_scaled:
        r,h = np.linspace(1e-12, r_cut+0.5, num_spline_points, retstep=True)
        A0_spline_values = []
        A0_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=torch.get_default_dtype()).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers)),
                    torch.tensor([[model_i],[model_j]], dtype=torch.int64),
                    model.atomic_numbers)
                if isinstance(bessels, tuple):
                    bessels = bessels[0]  # newer versions return (bessels, cutoffs)
                R = torch.tanh(model.interactions[0].density_fn(bessels)**2).numpy(force=True)
                spl = CubicSpline(r, R[:,0])
                A0_spline_values.append(spl(r).tolist())
                A0_spline_derivs.append(spl.derivative()(r).tolist())
        output['A0_spline_h'] = float(h)
        output['A0_spline_values'] = A0_spline_values
        output['A0_spline_derivs'] = A0_spline_derivs
    A0_weights = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        A0_weights.append([])
        for l,_,w in model.interactions[0].skip_tp.weight_views(yield_instruction=True):
            w_linear = model.interactions[0].linear.weight_view_for_instruction(l).numpy(force=True) / np.sqrt(num_channels)
            if not A0_scaled:
                w_linear /= model.interactions[0].avg_num_neighbors
            fused = w_linear @ w[:,model_i,:].numpy(force=True) / np.sqrt(len(model.atomic_numbers)*num_channels)
            A0_weights[i].append(fused.flatten().tolist())
    output["A0_weights"] = A0_weights

    #### ----- M0 -----

    logging.info("M0")
    correlation = model.products[0].symmetric_contractions.contractions[0].correlation
    ### Computes U_{lm\eta, l1m1 l2m2 ...}
    #   * `irrep_out` is essentially l_out
    #   * `irreps_in` essentially provides l_in_max
    #   * `corr_in_max` is the max correlation order
    def U_sparse(irrep_out, irreps_in, corr_in_max):
        U = [[]]  # list of lists because U[0] should be empty
        for corr in range(1,corr_in_max+1):
            # get U matrix for this correlation order
            try:
                U_matrix = U_matrix_real(irreps_in, [irrep_out], corr, use_cueq_cg=False)[1]
            except TypeError:
                U_matrix = U_matrix_real(irreps_in, [irrep_out], corr)[1]
            if irrep_out.l == 0:  # makes U_matrix.shape consistent with l>0 cases
                U_matrix = U_matrix.unsqueeze(0)
            num_eta = U_matrix.shape[-1]
            U_matrix = U_matrix.flatten()
            # extract sparse U for this correlation order
            U_sparse_corr = [[{} for _ in range(num_eta)] for _ in range(2*irrep_out.l+1)]
            j = 0
            for m in range(2*irrep_out.l+1):
                for lm_list in itertools.product(range((l_max+1)**2), repeat=corr):
                    for eta in range(num_eta):
                        if abs(U_matrix[j]) > 1e-12:
                            lm_tuple_sorted = tuple(sorted(lm_list))
                            if lm_tuple_sorted not in U_sparse_corr[m][eta].keys():
                                U_sparse_corr[m][eta][lm_tuple_sorted] = 0.0
                            U_sparse_corr[m][eta][lm_tuple_sorted] += U_matrix[j].item()
                        j += 1
            U.append(U_sparse_corr)
        return U
    irreps_in = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_in]
    irreps_out = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_out]
    C = {}
    M = {}
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        C[i] = {}
        for l, irrep_out in enumerate(irreps_out):
            # extract U in sparse format
            U = U_sparse(irrep_out, irreps_in, correlation)
            # extract weights from model
            # warning: slightly odd order of the contractions weights due to reverse countdown
            W = [[]]  # list of lists because W[0] should be empty
            W.append(model.products[0].symmetric_contractions.contractions[l].weights[1].numpy(force=True))
            W.append(model.products[0].symmetric_contractions.contractions[l].weights[0].numpy(force=True))
            W.append(model.products[0].symmetric_contractions.contractions[l].weights_max.numpy(force=True))
            # combine U and W into polynomial-like terms for recursive evaluator
            for m in range(-l,l+1):
                lm = l*(l+1)+m
                C[i][lm] = {}
                for k in range(num_channels):
                    P_lmk = {}
                    for corr in range(1,correlation+1):
                        for eta in range(len(U[corr][l+m])):
                            for key,value in U[corr][l+m][eta].items():
                                if key not in P_lmk.keys():
                                    P_lmk[key] = 0.0
                                P_lmk[key] += float(W[corr][model_i,eta,k]) * value
                    C[i][lm][k] = list(P_lmk.values())
                    M[lm] = [list(key) for key in P_lmk.keys()]
    output['M0_weights'] = C
    output['M0_monomials'] = M

    ### ----- H1 -----

    logging.info("H1")
    H1_weights = np.zeros([L_max+1, num_channels, num_channels])
    weights_0 = np.reshape(
        model.products[0].linear.weight.numpy(force=True),
        [L_max+1,num_channels,num_channels]) / np.sqrt(num_channels)
    weights_1 = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True),
        [L_max+1,num_channels,num_channels]) / np.sqrt(num_channels)
    for l in range(L_max+1):
        H1_weights[l,:,:] = weights_0[l,:,:] @ weights_1[l,:,:]
    output["H1_weights"] = H1_weights.flatten().tolist()

    ### ----- Phi1 -----

    logging.info("Phi1")
    Phi1_l = [ir[1].l for ir in model.interactions[1].conv_tp.irreps_out]
    Phi1_l1 = [ins.i_in2 for ins in model.interactions[1].conv_tp.instructions]
    Phi1_l2 = [ins.i_in1 for ins in model.interactions[1].conv_tp.instructions]
    Phi1_clebsch_gordan = []
    Phi1_lme = []
    Phi1_lelm1lm2 = []
    num_lm1 = (l_max+1)**2
    num_lm2 = (L_max+1)**2
    def compute_lem(le, l, m):
        lem = 0
        for j in range(le):
            lem += 2*Phi1_l[j]+1
        return lem+l+m
    def compute_lme(le, l, m):
        e = le - int(sum(np.array(Phi1_l)<l))
        num_e = [int(sum(np.array(Phi1_l)==ll)) for ll in range(l+1)]
        lme = 0
        for ll in range(l):
            lme += (2*ll+1)*num_e[ll]
        return lme + (l+m)*num_e[l]+e
    def compute_lelm1lm2(le,l1,m1,l2,m2):
        lelm1lm2 = 0
        for j in range(le):
            l1,l2 = (Phi1_l1[j], Phi1_l2[j])
            lelm1lm2 += (2*l1+1)*(2*l2+1)
        l, l1, l2 = (Phi1_l[le], Phi1_l1[le], Phi1_l2[le])
        return lelm1lm2 + (l1+m1)*(2*l2+1) + l2+m2
    tp = model.interactions[1].conv_tp
    for l1 in range(l_max+1):
        for m1 in range(-l1,l1+1):
            lm1 = l1*l1+l1+m1
            for l2 in range(L_max+1):
                for m2 in range(-l2,l2+1):
                    R = torch.ones([1,len(tp.instructions)*num_channels],dtype=torch.double)
                    Y = torch.zeros([1,num_lm1], dtype=torch.double)
                    Y[0,lm1] = 1.0
                    H = torch.zeros([1,num_lm2*num_channels],dtype=torch.double)
                    H[0,sum([2*p+1 for p in range(l2)])*num_channels+l2+m2] = 1.0
                    Phi = tp(H, Y, R)
                    # extract Phi values for k=0
                    Phi_0 = []
                    for le in range(len(tp.instructions)):
                        Phi_0_start = sum([2*Phi1_l[p]+1 for p in range(le)])*num_channels
                        for p in range(2*Phi1_l[le]+1):
                            Phi_0.append(Phi[0,Phi_0_start+p].item())
                    for le in range(len(tp.instructions)):
                        l = Phi1_l[le]
                        for m in range(-l,l+1):
                            lem = compute_lem(le,l,m)
                            if np.abs(Phi_0[lem]) > 1e-12:
                                Phi1_lme.append(compute_lme(le,l,m))
                                Phi1_clebsch_gordan.append(Phi_0[lem])
                                Phi1_lelm1lm2.append(compute_lelm1lm2(le,l1,m1,l2,m2))
    output["Phi1_l"] = Phi1_l
    output["Phi1_l1"] = Phi1_l1
    output["Phi1_l2"] = Phi1_l2
    output["Phi1_lme"] = Phi1_lme
    output["Phi1_clebsch_gordan"] = Phi1_clebsch_gordan
    output["Phi1_lelm1lm2"] = Phi1_lelm1lm2

    ### ----- A1 -----

    logging.info("A1")
    A1_scaled = True if ("Density" in type(model.interactions[1]).__name__) else False
    output['A1_scaled'] = A1_scaled
    if A1_scaled:
        r,h = np.linspace(1e-12, r_cut+0.5, num_spline_points, retstep=True)
        A1_spline_values = []
        A1_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels = model.radial_embedding(
                    torch.tensor(r, dtype=torch.get_default_dtype()).unsqueeze(-1),
                    torch.eye(len(model.atomic_numbers)),
                    torch.tensor([[model_i],[model_j]], dtype=torch.int64),
                    model.atomic_numbers)
                if isinstance(bessels, tuple):
                    bessels = bessels[0]  # newer versions return (bessels, cutoffs)
                R = torch.tanh(model.interactions[1].density_fn(bessels)**2).numpy(force=True)
                spl = CubicSpline(r, R[:,0])
                A1_spline_values.append(spl(r).tolist())
                A1_spline_derivs.append(spl.derivative()(r).tolist())
        output['A1_spline_h'] = float(h)
        output['A1_spline_values'] = A1_spline_values
        output['A1_spline_derivs'] = A1_spline_derivs
    A1_weights = []
    num_eta = [sum([l==ll for ll in Phi1_l]) for l in range(l_max+1)]
    A1_linear = linear_simplify(model.interactions[1].linear)
    for l in range(l_max+1):
        w_linear = A1_linear.weight_view_for_instruction(l).numpy(force=True) / np.sqrt(num_eta[l]*num_channels)
        if not A1_scaled:
            w_linear /= model.interactions[1].avg_num_neighbors
        w_linear = np.reshape(w_linear, (num_eta[l], num_channels, num_channels))
        A1_weights.append(w_linear.flatten().tolist())
    output["A1_weights"] = A1_weights

    ### ----- M1 -----

    logging.info("M1")
    # TODO: generalize
    correlation = model.products[1].symmetric_contractions.contractions[0].correlation
    irreps_in = Irreps("0e + 1o + 2e + 3o")
    irreps_out = Irreps("0e")
    # extract U in sparse format
    U = [[]]
    for corr in range(1,4):
        # get U matrix for this correlation order
        try:
            U_matrix = U_matrix_real(irreps_in, irreps_out, corr, use_cueq_cg=False)[1]
        except TypeError:
            U_matrix = U_matrix_real(irreps_in, irreps_out, corr)[1]
        num_nu = U_matrix.shape[-1]
        U_matrix = U_matrix.flatten()
        # extract sparse U for this correlation order
        U_sparse = [{} for _ in range(num_nu)]
        j = 0
        for lm_list in itertools.product(range((l_max+1)**2), repeat=corr):
            for nu in range(num_nu):
                if abs(U_matrix[j]) > 1e-12:
                    lm_tuple_sorted = tuple(sorted(lm_list))
                    if lm_tuple_sorted not in U_sparse[nu].keys():
                        U_sparse[nu][lm_tuple_sorted] = 0.0
                    U_sparse[nu][lm_tuple_sorted] += U_matrix[j].item()
                j += 1
        U.append(U_sparse)
    # extract weights from model
    # warning: slightly odd order of the contractions weights due to reverse countdown
    W = [[]]
    W.append(model.products[1].symmetric_contractions.contractions[0].weights[1].numpy(force=True))
    W.append(model.products[1].symmetric_contractions.contractions[0].weights[0].numpy(force=True))
    W.append(model.products[1].symmetric_contractions.contractions[0].weights_max.numpy(force=True))
    # combine U and W into polynomial-like terms for recursive evaluator
    C = {}
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        C[i] = {}
        for k in range(num_channels):
            P_ik = {}
            for corr in range(1,4):
                for nu in range(len(U[corr])):
                    for key,value in U[corr][nu].items():
                        if key not in P_ik.keys():
                            P_ik[key] = 0.0
                        P_ik[key] += float(W[corr][model_i,nu,k]) * value
            C[i][k] = list(P_ik.values())
            M = [list(key) for key in P_ik.keys()]
    output['M1_weights'] = C
    output['M1_monomials'] = M

    ### ----- H2 -----

    logging.info("H2")
    weights_to_fuse = model.interactions[1].linear_up.weight_view_for_instruction(0).numpy(force=True) / np.sqrt(num_channels)
    weights_to_fuse_rank = np.linalg.matrix_rank(weights_to_fuse)
    if weights_to_fuse_rank < num_channels:
        raise RuntimeError('ERROR: fusing weights have too low rank {weights_to_fuse_rank} < {num_channels}')
    # H2 weights for H1
    H2_weights_for_H1 = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        w = model.interactions[1].skip_tp.weight_view_for_instruction(0)[:,model_i,:].numpy(force=True)
        H2_weights_for_H1.append(w / np.sqrt(len(model.atomic_numbers)*num_channels))
        H2_weights_for_H1[i] = np.linalg.inv(weights_to_fuse) @ H2_weights_for_H1[i]
        H2_weights_for_H1[i] = H2_weights_for_H1[i].flatten().tolist()
    output["H2_weights_for_H1"] = H2_weights_for_H1
    # H2 weights for M1
    output["H2_weights_for_M1"] = (
        model.products[1].linear.weight.numpy(force=True) / np.sqrt(num_channels)).tolist()

    ### ----- READOUTS -----

    # linear readout
    weights_to_fuse = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True) / np.sqrt(num_channels),
        [L_max+1,num_channels,num_channels])
    readout_1_weights = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(num_channels)
    readout_1_weights = np.linalg.inv(weights_to_fuse[0,:,:]) @ readout_1_weights
    output['readout_1_weights'] = (readout_1_weights * model.scale_shift.scale.item()).tolist()

    # nonlinear readout
    #output["mlp_hidden_layers"] = 16 // TODO
    output["readout_2_weights_1"] = (
            torch.reshape(
                model.readouts[1].linear_1.weight,
                (num_channels,16)
            ).T.numpy(force=True).flatten() / np.sqrt(num_channels)
        ).tolist()
    output["readout_2_weights_2"] = (
        model.readouts[1].linear_2.weight.numpy(force=True).flatten() / np.sqrt(16)
        * model.scale_shift.scale.item()).tolist()
    output["readout_2_scale_factor"] = model.readouts[1].non_linearity.acts[0].cst

    return output
