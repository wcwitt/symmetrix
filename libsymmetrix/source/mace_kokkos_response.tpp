template <typename Precision>
void MACEKokkos<Precision>::compute_electric_field_hessian(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field)
{
    if (!has_field_coupling)
        throw std::invalid_argument(
            "MACEKokkos::compute_electric_field_hessian requires field coupling.");
    if (electric_field.size() != 3)
        throw std::invalid_argument(
            "MACEKokkos::compute_electric_field_hessian requires a graph-level electric field.");
    if (L_max != 1 || num_LM != 4)
        throw std::invalid_argument(
            "MACEKokkos analytic field response currently requires L_max == 1.");

    // Keep energy, force, and polarization state at the requested field while
    // replacing the three perturbed full evaluations with an analytic pass.
    compute_node_energies_forces_field(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
        xyz, r, electric_field);

    if (electric_field_hessian.size() != 9)
        Kokkos::realloc(electric_field_hessian, 9);
    if (electric_field_force_derivative.size() != 3*xyz.size())
        Kokkos::realloc(electric_field_force_derivative, 3*xyz.size());
    Kokkos::deep_copy(electric_field_hessian, 0.0);
    Kokkos::deep_copy(electric_field_force_derivative, 0.0);

    Kokkos::View<int*> first_neigh("MACEField response first_neigh", num_nodes);
    Kokkos::parallel_scan(
        "MACEField response first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });
    Kokkos::fence();

    const int channels = num_channels;
    const int lm_count = num_lm;
    const int LM_count = num_LM;
    const int lmax = l_max;
    const int Lmax = L_max;
    const int phi_rows = num_lelm1lm2;
    const int phi_outputs = num_lme;
    const auto response_hessian = electric_field_hessian;
    const auto response_forces = electric_field_force_derivative;
    const int polynomial_nodes = M1_poly_coeff.extent(1);

    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_dot(
        "MACEField H1_dot", num_nodes, LM_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1r_dot(
        "MACEField Phi1r_dot", num_nodes, phi_rows, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1_dot(
        "MACEField Phi1_dot", num_nodes, phi_outputs, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> A1_dot(
        "MACEField A1_dot", num_nodes, lm_count, channels);
    Kokkos::View<Precision**,Kokkos::LayoutRight> M1_dot(
        "MACEField M1_dot", num_nodes, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> M1_value_dot(
        "MACEField M1 value_dot", num_nodes, polynomial_nodes, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> M1_gradient(
        "MACEField M1 gradient", num_nodes, polynomial_nodes, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> M1_gradient_dot(
        "MACEField M1 gradient_dot", num_nodes, polynomial_nodes, channels);
    Kokkos::View<double**,Kokkos::LayoutRight> H2_dot(
        "MACEField H2_dot", num_nodes, channels);
    Kokkos::View<double*> readout_output(
        "MACEField directional readout", num_nodes);
    Kokkos::View<double**,Kokkos::LayoutRight> H2_adj_local(
        "MACEField H2_adj", num_nodes, channels);
    Kokkos::View<double**,Kokkos::LayoutRight> H2_adj_dot(
        "MACEField H2_adj_dot", num_nodes, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj_local(
        "MACEField H1_adj", num_nodes, LM_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj_dot(
        "MACEField H1_adj_dot", num_nodes, LM_count, channels);
    Kokkos::View<Precision**,Kokkos::LayoutRight> M1_adj_local(
        "MACEField M1_adj", num_nodes, channels);
    Kokkos::View<Precision**,Kokkos::LayoutRight> M1_adj_dot(
        "MACEField M1_adj_dot", num_nodes, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> A1_adj_local(
        "MACEField A1_adj", num_nodes, lm_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> A1_adj_dot(
        "MACEField A1_adj_dot", num_nodes, lm_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1_local(
        "MACEField dPhi1", num_nodes, phi_outputs, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1_dot(
        "MACEField dPhi1_dot", num_nodes, phi_outputs, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1r_local(
        "MACEField dPhi1r", num_nodes, phi_rows, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1r_dot(
        "MACEField dPhi1r_dot", num_nodes, phi_rows, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_product_adj_dot(
        "MACEField product adjoint tangent", num_nodes, LM_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> M0_adj_dot(
        "MACEField M0 adjoint tangent", num_nodes, LM_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> A0_adj_dot(
        "MACEField A0 adjoint tangent", num_nodes, lm_count, channels);

    for (int seed=0; seed<3; ++seed) {
        const auto scalar_to_vector_up = field_scalar_to_vector_up_matrix;
        const auto vector_to_scalar_up = field_vector_to_scalar_up_matrix;
        const auto H1_before_field = H1_pre_field;
        const auto H1_up_weights = H1_linear_up_weights;
        Kokkos::parallel_for(
            "MACEField analytic field and linear-up tangent",
            Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Right>>(
                {0,0,0}, {num_nodes,LM_count,channels}),
            KOKKOS_LAMBDA (const int i, const int lm, const int output) {
                Precision value = 0.0;
                if (lm == 0) {
                    for (int input=0; input<channels; ++input)
                        value += vector_to_scalar_up(input,output)
                            *H1_before_field(i,1+seed,input);
                } else if (lm == 1+seed) {
                    for (int input=0; input<channels; ++input)
                        value += scalar_to_vector_up(input,output)
                            *H1_before_field(i,0,input);
                }
                H1_dot(i,lm,output) = value;
            });

        Kokkos::deep_copy(Phi1_dot, 0.0);
        const auto phi_lm1 = Phi1_lm1;
        const auto phi_lm2 = Phi1_lm2;
        const auto phi_path = Phi1_lel1l2;
        const auto radial1 = R1;
        const auto harmonics = Y;
        Kokkos::parallel_for(
            "MACEField analytic Phi1r tangent",
            Kokkos::TeamPolicy<>(num_nodes*phi_rows, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/phi_rows;
                const int row = team.league_rank()%phi_rows;
                const int edge_begin = first_neigh(i);
                const int lm1 = phi_lm1(row);
                const int lm2 = phi_lm2(row);
                const int path = phi_path(row);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, channels),
                    [=] (const int channel) {
                        Precision value = 0.0;
                        for (int j=0; j<num_neigh(i); ++j) {
                            const int edge = edge_begin+j;
                            value += radial1(edge,path*channels+channel)
                                *harmonics(edge*lm_count+lm1)
                                *H1_dot(neigh_indices(edge),lm2,channel);
                        }
                        Phi1r_dot(i,row,channel) = value;
                    });
            });

        const auto phi_lme = Phi1_lme;
        const auto phi_row = Phi1_lelm1lm2;
        const auto phi_cg = Phi1_clebsch_gordan;
        Kokkos::parallel_for(
            "MACEField analytic Phi1 tangent",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                for (int p=0; p<phi_cg.extent(0); ++p) {
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, channels),
                        [=] (const int channel) {
                            Phi1_dot(i,phi_lme(p),channel) += phi_cg(p)
                                *Phi1r_dot(i,phi_row(p),channel);
                        });
                }
            });

        const auto phi_l = Phi1_l;
        const auto A1_matrix = A1_weights;
        Kokkos::parallel_for(
            "MACEField analytic A1 tangent",
            Kokkos::TeamPolicy<>(num_nodes*(lmax+1), Kokkos::AUTO),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/(lmax+1);
                const int l = team.league_rank()%(lmax+1);
                int lme = 0;
                int multiplicity = 0;
                for (int p=0; p<phi_l.extent(0); ++p) {
                    if (phi_l(p) < l)
                        lme += 2*phi_l(p)+1;
                    if (phi_l(p) == l)
                        ++multiplicity;
                }
                auto input = Kokkos::View<Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>(
                        &Phi1_dot(i,lme,0), 2*l+1, multiplicity*channels);
                auto output = Kokkos::subview(
                    A1_dot, i, Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Gemm::Blocked>::invoke(
                        team, 1.0, input, A1_matrix(l), 0.0, output);
            });

        const auto A1_scale_values = A1_spline_values;
        if (A1_scaled) {
            Kokkos::parallel_for(
                "MACEField analytic A1 tangent scaling",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    double scale;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j, double& sum) {
                            sum += A1_scale_values(first_neigh(i)+j,0);
                        }, scale);
                    scale += 1.0;
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index) {
                            A1_dot(i,index/channels,index%channels) /= scale;
                        });
                });
        }

        Kokkos::deep_copy(M1_dot, 0.0);
        const auto M1_values = M1_poly_values;
        const auto M1_spec = M1_poly_spec;
        const auto M1_coeff = M1_poly_coeff;
        Kokkos::parallel_for(
            "MACEField analytic M1 directional graph",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamVectorMDRange<
                        Kokkos::Rank<2,Kokkos::Iterate::Right>,
                        Kokkos::TeamPolicy<>::member_type>(
                            team, lm_count, channels),
                    [=] (const int lm, const int channel) {
                        M1_value_dot(i,lm,channel) = A1_dot(i,lm,channel);
                        Kokkos::atomic_add(
                            &M1_dot(i,channel),
                            M1_coeff(node_types(i),lm,channel)
                                *M1_value_dot(i,lm,channel));
                    });
                team.team_barrier();
                for (int p=0; p<M1_spec.extent(0); ++p) {
                    const int p0 = M1_spec(p,0);
                    const int p1 = M1_spec(p,1);
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, channels),
                        [=] (const int channel) {
                            const int node = lm_count+p;
                            M1_value_dot(i,node,channel) =
                                M1_value_dot(i,p0,channel)*M1_values(i,p1,channel)
                                +M1_values(i,p0,channel)*M1_value_dot(i,p1,channel);
                            M1_dot(i,channel) +=
                                M1_coeff(node_types(i),node,channel)
                                *M1_value_dot(i,node,channel);
                        });
                }
                team.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::TeamVectorMDRange<
                        Kokkos::Rank<2,Kokkos::Iterate::Right>,
                        Kokkos::TeamPolicy<>::member_type>(
                            team, polynomial_nodes, channels),
                    [=] (const int node, const int channel) {
                        M1_gradient(i,node,channel) =
                            M1_coeff(node_types(i),node,channel);
                        M1_gradient_dot(i,node,channel) = 0.0;
                    });
                team.team_barrier();
                for (int p=M1_spec.extent(0)-1; p>=0; --p) {
                    const int p0 = M1_spec(p,0);
                    const int p1 = M1_spec(p,1);
                    const int node = lm_count+p;
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, channels),
                        [=] (const int channel) {
                            const Precision adjoint = M1_gradient(i,node,channel);
                            const Precision adjoint_dot =
                                M1_gradient_dot(i,node,channel);
                            M1_gradient_dot(i,p0,channel) +=
                                adjoint_dot*M1_values(i,p1,channel)
                                +adjoint*M1_value_dot(i,p1,channel);
                            M1_gradient_dot(i,p1,channel) +=
                                adjoint_dot*M1_values(i,p0,channel)
                                +adjoint*M1_value_dot(i,p0,channel);
                            M1_gradient(i,p0,channel) +=
                                adjoint*M1_values(i,p1,channel);
                            M1_gradient(i,p1,channel) +=
                                adjoint*M1_values(i,p0,channel);
                        });
                }
            });

        const auto H2_from_H1 = H2_weights_for_H1;
        const auto H2_from_M1 = H2_weights_for_M1;
        Kokkos::parallel_for(
            "MACEField analytic H2 tangent",
            Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Right>>(
                {0,0}, {num_nodes,channels}),
            KOKKOS_LAMBDA (const int i, const int output) {
                double value = 0.0;
                for (int input=0; input<channels; ++input) {
                    value += H2_from_H1(node_types(i),input*channels+output)
                        *static_cast<double>(H1_dot(i,0,input));
                    value += H2_from_M1(input*channels+output)
                        *static_cast<double>(M1_dot(i,input));
                }
                H2_dot(i,output) = value;
            });

        auto H2_input = Kokkos::subview(H2, Kokkos::make_pair(0,num_nodes), Kokkos::ALL);
        readout_2.evaluate_gradient_directional(
            H2_input, H2_dot, readout_output, H2_adj_local, H2_adj_dot);

        Kokkos::deep_copy(H1_adj_local, 0.0);
        Kokkos::deep_copy(H1_adj_dot, 0.0);
        const auto readout1 = readout_1_weights;
        Kokkos::parallel_for(
            "MACEField analytic H2 reverse tangent",
            Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Right>>(
                {0,0}, {num_nodes,channels}),
            KOKKOS_LAMBDA (const int i, const int input) {
                Precision h1_adjoint = static_cast<Precision>(readout1(input));
                Precision h1_adjoint_dot = 0.0;
                Precision m1_adjoint = 0.0;
                Precision m1_adjoint_dot = 0.0;
                for (int output=0; output<channels; ++output) {
                    const double h1_weight =
                        H2_from_H1(node_types(i),input*channels+output);
                    const double m1_weight = H2_from_M1(input*channels+output);
                    h1_adjoint += static_cast<Precision>(
                        h1_weight*H2_adj_local(i,output));
                    h1_adjoint_dot += static_cast<Precision>(
                        h1_weight*H2_adj_dot(i,output));
                    m1_adjoint += static_cast<Precision>(
                        m1_weight*H2_adj_local(i,output));
                    m1_adjoint_dot += static_cast<Precision>(
                        m1_weight*H2_adj_dot(i,output));
                }
                H1_adj_local(i,0,input) = h1_adjoint;
                H1_adj_dot(i,0,input) = h1_adjoint_dot;
                M1_adj_local(i,input) = m1_adjoint;
                M1_adj_dot(i,input) = m1_adjoint_dot;
            });

        Kokkos::parallel_for(
            "MACEField analytic M1 reverse tangent",
            Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Right>>(
                {0,0,0}, {num_nodes,lm_count,channels}),
            KOKKOS_LAMBDA (const int i, const int lm, const int channel) {
                const Precision gradient = M1_gradient(i,lm,channel);
                A1_adj_local(i,lm,channel) =
                    gradient*M1_adj_local(i,channel);
                A1_adj_dot(i,lm,channel) =
                    M1_gradient_dot(i,lm,channel)*M1_adj_local(i,channel)
                    +gradient*M1_adj_dot(i,channel);
            });

        const auto A1_base = A1;
        const auto A1_scale_derivs = A1_spline_derivs;
        if (A1_scaled) {
            Kokkos::parallel_for(
                "MACEField analytic A1 scaled reverse tangent",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    double scale;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j, double& sum) {
                            sum += A1_scale_values(first_neigh(i)+j,0);
                        }, scale);
                    scale += 1.0;
                    double contraction;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index, double& sum) {
                            const int lm = index/channels;
                            const int channel = index%channels;
                            sum += static_cast<double>(A1_adj_dot(i,lm,channel))
                                *static_cast<double>(A1_base(i,lm,channel));
                            sum += static_cast<double>(A1_adj_local(i,lm,channel))
                                *static_cast<double>(A1_dot(i,lm,channel));
                        }, contraction);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j) {
                            const int edge = first_neigh(i)+j;
                            const double factor = contraction/scale
                                *A1_scale_derivs(edge,0)/r(edge);
                            response_forces(seed*xyz.size()+3*edge) +=
                                factor*xyz(3*edge);
                            response_forces(seed*xyz.size()+3*edge+1) +=
                                factor*xyz(3*edge+1);
                            response_forces(seed*xyz.size()+3*edge+2) +=
                                factor*xyz(3*edge+2);
                        });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index) {
                            const int lm = index/channels;
                            const int channel = index%channels;
                            A1_adj_local(i,lm,channel) /= scale;
                            A1_adj_dot(i,lm,channel) /= scale;
                        });
                });
        }

        const auto A1_matrix_trans = A1_weights_trans;
        Kokkos::parallel_for(
            "MACEField analytic A1 reverse",
            Kokkos::TeamPolicy<>(num_nodes*(lmax+1), Kokkos::AUTO),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/(lmax+1);
                const int l = team.league_rank()%(lmax+1);
                int lme = 0;
                int multiplicity = 0;
                for (int p=0; p<phi_l.extent(0); ++p) {
                    if (phi_l(p) < l)
                        lme += 2*phi_l(p)+1;
                    if (phi_l(p) == l)
                        ++multiplicity;
                }
                auto base_input = Kokkos::subview(
                    A1_adj_local, i,
                    Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
                auto dot_input = Kokkos::subview(
                    A1_adj_dot, i,
                    Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
                auto base_output = Kokkos::View<Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>(
                        &dPhi1_local(i,lme,0), 2*l+1, multiplicity*channels);
                auto dot_output = Kokkos::View<Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>(
                        &dPhi1_dot(i,lme,0), 2*l+1, multiplicity*channels);
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Gemm::Blocked>::invoke(
                        team, 1.0, base_input, A1_matrix_trans(l),
                        0.0, base_output);
                team.team_barrier();
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Gemm::Blocked>::invoke(
                        team, 1.0, dot_input, A1_matrix_trans(l),
                        0.0, dot_output);
            });

        Kokkos::deep_copy(dPhi1r_local, 0.0);
        Kokkos::deep_copy(dPhi1r_dot, 0.0);
        Kokkos::parallel_for(
            "MACEField analytic CG reverse tangent",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                for (int p=0; p<phi_cg.extent(0); ++p) {
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, channels),
                        [=] (const int channel) {
                            const int row = phi_row(p);
                            const int output = phi_lme(p);
                            dPhi1r_local(i,row,channel) +=
                                phi_cg(p)*dPhi1_local(i,output,channel);
                            dPhi1r_dot(i,row,channel) +=
                                phi_cg(p)*dPhi1_dot(i,output,channel);
                        });
                }
            });

        const auto radial1_deriv = R1_deriv;
        const auto harmonics_grad = Y_grad;
        const auto H1_base = H1;
        {
            const Kokkos::TeamPolicy<> phi_reverse_policy(
                num_nodes, Kokkos::AUTO, 32);
            Kokkos::parallel_for(
                "MACEField analytic Phi1 reverse tangent",
                phi_reverse_policy,
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                const int edge_begin = first_neigh(i);
                for (int j=0; j<num_neigh(i); ++j) {
                    const int edge = edge_begin+j;
                    const int neighbor = neigh_indices(edge);
                    const double inverse_radius = 1.0/r(edge);
                    double force_x, force_y, force_z;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, phi_rows),
                        [=] (const int row,
                             double& local_x,
                             double& local_y,
                             double& local_z) {
                            const int lm1 = phi_lm1(row);
                            const int lm2 = phi_lm2(row);
                            const int path = phi_path(row);
                            const Precision y = harmonics(edge*lm_count+lm1);
                            double row_x, row_y, row_z;
                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team, channels),
                                [=] (const int channel,
                                     double& channel_x,
                                     double& channel_y,
                                     double& channel_z) {
                                    const Precision radial =
                                        radial1(edge,path*channels+channel);
                                    const Precision radial_derivative =
                                        radial1_deriv(edge,path*channels+channel);
                                    const Precision base_feature =
                                        H1_base(neighbor,lm2,channel);
                                    const Precision feature_dot =
                                        H1_dot(neighbor,lm2,channel);
                                    const Precision base_adjoint =
                                        dPhi1r_local(i,row,channel);
                                    const Precision adjoint_dot =
                                        dPhi1r_dot(i,row,channel);
                                    const double radial_base = static_cast<double>(
                                        radial_derivative*y*base_feature);
                                    const double radial_dot = static_cast<double>(
                                        radial_derivative*y*feature_dot);
                                    const double angular_base = static_cast<double>(
                                        radial*base_feature);
                                    const double angular_dot = static_cast<double>(
                                        radial*feature_dot);
                                    const double base_adj = static_cast<double>(base_adjoint);
                                    const double dot_adj = static_cast<double>(adjoint_dot);
                                    channel_x -= dot_adj*(
                                        radial_base*xyz(3*edge)*inverse_radius
                                        +angular_base*harmonics_grad(3*edge*lm_count+lm1));
                                    channel_x -= base_adj*(
                                        radial_dot*xyz(3*edge)*inverse_radius
                                        +angular_dot*harmonics_grad(3*edge*lm_count+lm1));
                                    channel_y -= dot_adj*(
                                        radial_base*xyz(3*edge+1)*inverse_radius
                                        +angular_base*harmonics_grad((3*edge+1)*lm_count+lm1));
                                    channel_y -= base_adj*(
                                        radial_dot*xyz(3*edge+1)*inverse_radius
                                        +angular_dot*harmonics_grad((3*edge+1)*lm_count+lm1));
                                    channel_z -= dot_adj*(
                                        radial_base*xyz(3*edge+2)*inverse_radius
                                        +angular_base*harmonics_grad((3*edge+2)*lm_count+lm1));
                                    channel_z -= base_adj*(
                                        radial_dot*xyz(3*edge+2)*inverse_radius
                                        +angular_dot*harmonics_grad((3*edge+2)*lm_count+lm1));
                                    const Precision source_factor = radial*y;
                                    Kokkos::atomic_add(
                                        &H1_adj_local(neighbor,lm2,channel),
                                        source_factor*base_adjoint);
                                    Kokkos::atomic_add(
                                        &H1_adj_dot(neighbor,lm2,channel),
                                        source_factor*adjoint_dot);
                                }, row_x, row_y, row_z);
                            local_x += row_x;
                            local_y += row_y;
                            local_z += row_z;
                        }, force_x, force_y, force_z);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        response_forces(seed*xyz.size()+3*edge) += force_x;
                        response_forces(seed*xyz.size()+3*edge+1) += force_y;
                        response_forces(seed*xyz.size()+3*edge+2) += force_z;
                    });
                }
                });
        }

        Kokkos::parallel_reduce(
            "MACEField analytic field reverse tangent",
            num_nodes*channels,
            AnalyticFieldReverseReducer<Precision>{
                channels,
                seed,
                H1_adj_local,
                H1_adj_dot,
                H1_before_field,
                H1_product_adj_dot,
                H1_up_weights,
                scalar_to_vector_up,
                vector_to_scalar_up,
                electric_field,
                response_hessian
            });

        const auto product_weights = H1_product_weights;
        Kokkos::parallel_for(
            "MACEField analytic H1 product reverse tangent",
            Kokkos::TeamPolicy<>(num_nodes*(Lmax+1), Kokkos::AUTO),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/(Lmax+1);
                const int l = team.league_rank()%(Lmax+1);
                auto input = Kokkos::subview(
                    H1_product_adj_dot, i,
                    Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                auto weights = Kokkos::subview(
                    product_weights, l, Kokkos::ALL, Kokkos::ALL);
                auto output = Kokkos::subview(
                    M0_adj_dot, i,
                    Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::Transpose,
                    KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                        team, 1.0, input, weights, 0.0, output);
            });

        Kokkos::deep_copy(A0_adj_dot, 0.0);
        const auto M0_specs = M0_poly_spec;
        const auto M0_coeffs = M0_poly_coeff;
        const auto M0_values = M0_poly_values;
        for (int LM=0; LM<LM_count; ++LM) {
            const auto spec = M0_specs(LM);
            const auto coefficients = M0_coeffs(LM);
            const auto values = M0_values(LM);
            const int graph_nodes = coefficients.extent(1);
            Kokkos::View<Precision***,Kokkos::LayoutRight> graph_adj_dot(
                "MACEField M0 graph adjoint tangent",
                num_nodes, graph_nodes, channels);
            Kokkos::parallel_for(
                "MACEField analytic M0 reverse tangent",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorMDRange<
                            Kokkos::Rank<2,Kokkos::Iterate::Right>,
                            Kokkos::TeamPolicy<>::member_type>(
                                team, graph_nodes, channels),
                        [=] (const int node, const int channel) {
                            graph_adj_dot(i,node,channel) =
                                coefficients(node_types(i),node,channel)
                                *M0_adj_dot(i,LM,channel);
                        });
                    team.team_barrier();
                    for (int p=spec.extent(0)-1; p>=0; --p) {
                        const int p0 = spec(p,0);
                        const int p1 = spec(p,1);
                        const int node = lm_count+p;
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(team, channels),
                            [=] (const int channel) {
                                const Precision adjoint =
                                    graph_adj_dot(i,node,channel);
                                graph_adj_dot(i,p0,channel) +=
                                    adjoint*values(i,p1,channel);
                                graph_adj_dot(i,p1,channel) +=
                                    adjoint*values(i,p0,channel);
                            });
                    }
                    team.team_barrier();
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorMDRange<
                            Kokkos::Rank<2,Kokkos::Iterate::Right>,
                            Kokkos::TeamPolicy<>::member_type>(
                                team, lm_count, channels),
                        [=] (const int lm, const int channel) {
                            Kokkos::atomic_add(
                                &A0_adj_dot(i,lm,channel),
                                graph_adj_dot(i,lm,channel));
                        });
                });
        }

        const auto A0_base = A0;
        const auto A0_scale_values = A0_spline_values;
        const auto A0_scale_derivs = A0_spline_derivs;
        if (A0_scaled) {
            Kokkos::parallel_for(
                "MACEField analytic A0 scaled reverse tangent",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    double scale;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j, double& sum) {
                            sum += A0_scale_values(first_neigh(i)+j,0);
                        }, scale);
                    scale += 1.0;
                    double contraction;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index, double& sum) {
                            const int lm = index/channels;
                            const int channel = index%channels;
                            sum += static_cast<double>(A0_adj_dot(i,lm,channel))
                                *static_cast<double>(A0_base(i,lm,channel));
                        }, contraction);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j) {
                            const int edge = first_neigh(i)+j;
                            const double factor = contraction/scale
                                *A0_scale_derivs(edge,0)/r(edge);
                            response_forces(seed*xyz.size()+3*edge) +=
                                factor*xyz(3*edge);
                            response_forces(seed*xyz.size()+3*edge+1) +=
                                factor*xyz(3*edge+1);
                            response_forces(seed*xyz.size()+3*edge+2) +=
                                factor*xyz(3*edge+2);
                        });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index) {
                            A0_adj_dot(i,index/channels,index%channels) /= scale;
                        });
                });
        }

        const auto radial0 = R0;
        const auto radial0_deriv = R0_deriv;
        {
            const Kokkos::TeamPolicy<> A0_reverse_policy(
                num_nodes, Kokkos::AUTO, 32);
            Kokkos::parallel_for(
                "MACEField analytic A0 reverse tangent",
                A0_reverse_policy,
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                const int edge_begin = first_neigh(i);
                for (int j=0; j<num_neigh(i); ++j) {
                    const int edge = edge_begin+j;
                    const double inverse_radius = 1.0/r(edge);
                    double force_x, force_y, force_z;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, lmax+1),
                        [=] (const int l,
                             double& local_x,
                             double& local_y,
                             double& local_z) {
                            double l_x, l_y, l_z;
                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team, channels),
                                [=] (const int channel,
                                     double& channel_x,
                                     double& channel_y,
                                     double& channel_z) {
                                    for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                                        const double adjoint = static_cast<double>(
                                            A0_adj_dot(i,lm,channel));
                                        const double radial = static_cast<double>(
                                            radial0_deriv(edge,l*channels+channel))
                                            *static_cast<double>(
                                                harmonics(edge*lm_count+lm))
                                            *adjoint;
                                        const double angular = static_cast<double>(
                                            radial0(edge,l*channels+channel))*adjoint;
                                        channel_x -= radial*xyz(3*edge)*inverse_radius
                                            +angular*harmonics_grad(3*edge*lm_count+lm);
                                        channel_y -= radial*xyz(3*edge+1)*inverse_radius
                                            +angular*harmonics_grad((3*edge+1)*lm_count+lm);
                                        channel_z -= radial*xyz(3*edge+2)*inverse_radius
                                            +angular*harmonics_grad((3*edge+2)*lm_count+lm);
                                    }
                                }, l_x, l_y, l_z);
                            local_x += l_x;
                            local_y += l_y;
                            local_z += l_z;
                        }, force_x, force_y, force_z);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        response_forces(seed*xyz.size()+3*edge) += force_x;
                        response_forces(seed*xyz.size()+3*edge+1) += force_y;
                        response_forces(seed*xyz.size()+3*edge+2) += force_z;
                    });
                }
                });
        }
        Kokkos::fence();
    }

    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_electric_field_force_derivative(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field)
{
    compute_electric_field_hessian(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
        xyz, r, electric_field);
}
