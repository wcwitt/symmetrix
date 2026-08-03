#include <stdexcept>

#include "KokkosBlas.hpp"

#include "multilayer_perceptron_kokkos.hpp"

MultilayerPerceptronKokkos::MultilayerPerceptronKokkos()
{
    // TODO: add sanity checks to default constructor
}

MultilayerPerceptronKokkos::MultilayerPerceptronKokkos(
    std::vector<int> shape,
    std::vector<std::vector<double>> weights,
    double activation_scale)
{
    // TODO: currently this->shape is in SharedSpace ... is this okay?
    Kokkos::realloc(this->shape, shape.size());
    for (int i=0; i<shape.size(); ++i)
        this->shape(i) = shape[i];

    this->weights = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("weights", Kokkos::SequentialHostInit), weights.size());
    for (int l=0; l<shape.size()-1; ++l) {
        Kokkos::realloc(this->weights(l), shape[l+1], shape[l]);
        auto h_w = Kokkos::create_mirror_view(this->weights(l));
        for (int i=0; i<shape[l+1]; ++i)
            for (int j=0; j<shape[l]; ++j)
                h_w(i,j) = weights[l][i*shape[l]+j];
        Kokkos::deep_copy(this->weights(l), h_w);
    }

    this->activation_scale = activation_scale;

    this->node_values = Kokkos::View<
        Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
            Kokkos::view_alloc("node_values", Kokkos::SequentialHostInit), shape.size());

    this->node_derivatives = Kokkos::View<
        Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
            Kokkos::view_alloc("node_derivatives", Kokkos::SequentialHostInit), shape.size());

    this->node_value_dots = Kokkos::View<
        Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
            Kokkos::view_alloc("node_value_dots", Kokkos::SequentialHostInit), shape.size());

    this->node_derivative_dots = Kokkos::View<
        Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
            Kokkos::view_alloc("node_derivative_dots", Kokkos::SequentialHostInit), shape.size());
}

MultilayerPerceptronKokkos::~MultilayerPerceptronKokkos()
{
    Kokkos::fence();

    weights = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    node_values = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    node_derivatives = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    node_value_dots = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    node_derivative_dots = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
}

void MultilayerPerceptronKokkos::evaluate(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f)
{
    const int batch_size = x.extent(0);
    for (int l=0; l<shape.size(); ++l) {
        if (node_values(l).extent(0) != batch_size or node_values(l).extent(1) != shape(l))
            Kokkos::realloc(node_values(l), batch_size, shape(l));
        if (node_derivatives(l).extent(0) != batch_size or node_derivatives(l).extent(1) != shape(l))
            Kokkos::realloc(node_derivatives(l), batch_size, shape(l));
    }

    Kokkos::deep_copy(node_values(0), x);

    for (int i=0; i<batch_size; ++i) {
        for (int l=0; l<shape.size()-1; ++l) {
            auto in = Kokkos::subview(node_values(l), i, Kokkos::ALL);
            auto out = Kokkos::subview(node_values(l+1), i, Kokkos::ALL);
            KokkosBlas::SerialGemv<KokkosBatched::Trans::NoTranspose,
                                   KokkosBatched::Algo::Gemv::Unblocked>
                ::invoke(1.0, weights(l), in, 0.0, out);
            if (l==shape.size()-2)
                break;  // no activation in final layer
            Kokkos::parallel_for(
                "MultilayerPerceptronKokkos::evaluate::activation",
                shape(l+1),
                KOKKOS_CLASS_LAMBDA (const int j) {
                    out(j) = activation_scale*out(j)/(1.0+Kokkos::exp(-out(j)));
                });
        }
    }

    Kokkos::fence();
    Kokkos::deep_copy(f, Kokkos::subview(node_values(shape.size()-1), Kokkos::ALL, 0));
}

void MultilayerPerceptronKokkos::evaluate_gradient(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g)
{
    const int batch_size = x.extent(0);
    for (int l=0; l<shape.size(); ++l) {
        if (node_values(l).extent(0) != batch_size or node_values(l).extent(1) != shape(l))
            Kokkos::realloc(Kokkos::WithoutInitializing, node_values(l), batch_size, shape(l));
        Kokkos::deep_copy(node_values(l), 0.0);
        if (node_derivatives(l).extent(0) != batch_size or node_derivatives(l).extent(1) != shape(l))
            Kokkos::realloc(Kokkos::WithoutInitializing, node_derivatives(l), batch_size, shape(l));
        Kokkos::deep_copy(node_derivatives(l), 0.0);
    }

    Kokkos::deep_copy(node_values(0), x);

    auto activation_scale = this->activation_scale;
    auto node_derivatives = this->node_derivatives;
    auto node_values = this->node_values;
    auto shape = this->shape;
    auto weights = this->weights;

    Kokkos::parallel_for(
        "MultilayerPerceptronKokkos::evaluate_gradient",
        Kokkos::TeamPolicy<>(batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            // ----- forward -----
            for (int l=0; l<shape.size()-1; ++l) {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, weights(l).extent(0)),
                    [=] (const int p) {
                        for (int q=0; q<weights(l).extent(1); ++q)
                            node_values(l+1)(i,p) += weights(l)(p,q) * node_values(l)(i,q);
                    });
                team_member.team_barrier();
                if (l==shape.size()-2)
                    break;  // no activation in final layer
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, shape(l+1)),
                    [=] (const int j) {
                        node_values(l+1)(i,j) = activation_scale*node_values(l+1)(i,j)/(1.0+Kokkos::exp(-node_values(l+1)(i,j)));
                    });
                team_member.team_barrier();
            }
            // ----- reverse -----
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, node_derivatives(shape.size()-2).extent(1)),
                [=] (const int p) {
                    node_derivatives(shape.size()-2)(i,p) = weights(shape.size()-2)(0,p);
                });
            team_member.team_barrier();

            for (int l=shape.size()-3; l>=0; --l) {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, weights(l).extent(0)),
                    [=] (const int p) {
                        node_values(l+1)(i,p) = 0.0;
                        for (int q=0; q<weights(l).extent(1); ++q)
                            node_values(l+1)(i,p) += weights(l)(p,q) * node_values(l)(i,q);
                    });
                team_member.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, shape(l+1)),
                    [=] (const int j) {
                        const double x = node_values(l+1)(i,j);
                        const double sigmoid = 1.0/(1.0+std::exp(-x));
                        node_derivatives(l+1)(i,j) *= activation_scale*sigmoid
                            + activation_scale*x*sigmoid*(1-sigmoid);
                    });
                team_member.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, weights(l).extent(1)),
                    [=] (const int p) {
                        node_derivatives(l)(i,p) = 0.0;
                        for (int q=0; q<weights(l).extent(0); ++q)
                        node_derivatives(l)(i,p) += weights(l)(q,p) * node_derivatives(l+1)(i,q);
                    });
                team_member.team_barrier();
            }
        });
    Kokkos::fence();
    Kokkos::deep_copy(f, Kokkos::subview(node_values(shape.size()-1), Kokkos::ALL, 0));
    Kokkos::deep_copy(g, node_derivatives(0));
}

void MultilayerPerceptronKokkos::evaluate_gradient_directional(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot)
{
    const int batch_size = x.extent(0);
    if (x_dot.extent(0) != x.extent(0) || x_dot.extent(1) != x.extent(1))
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos directional input shape mismatch.");
    if (shape(shape.size()-1) != 1)
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos directional gradients require scalar output.");

    for (int l=0; l<shape.size(); ++l) {
        if (node_values(l).extent(0) != batch_size
            || node_values(l).extent(1) != shape(l))
            Kokkos::realloc(Kokkos::WithoutInitializing,
                            node_values(l), batch_size, shape(l));
        if (node_derivatives(l).extent(0) != batch_size
            || node_derivatives(l).extent(1) != shape(l))
            Kokkos::realloc(Kokkos::WithoutInitializing,
                            node_derivatives(l), batch_size, shape(l));
        if (node_value_dots(l).extent(0) != batch_size
            || node_value_dots(l).extent(1) != shape(l))
            Kokkos::realloc(Kokkos::WithoutInitializing,
                            node_value_dots(l), batch_size, shape(l));
        if (node_derivative_dots(l).extent(0) != batch_size
            || node_derivative_dots(l).extent(1) != shape(l))
            Kokkos::realloc(Kokkos::WithoutInitializing,
                            node_derivative_dots(l), batch_size, shape(l));
        Kokkos::deep_copy(node_derivatives(l), 0.0);
        Kokkos::deep_copy(node_derivative_dots(l), 0.0);
    }
    Kokkos::deep_copy(node_values(0), x);
    Kokkos::deep_copy(node_value_dots(0), x_dot);

    const auto activation_scale = this->activation_scale;
    const auto shape = this->shape;
    const auto weights = this->weights;
    const auto values = this->node_values;
    const auto value_dots = this->node_value_dots;
    const auto derivatives = this->node_derivatives;
    const auto derivative_dots = this->node_derivative_dots;

    Kokkos::parallel_for(
        "MultilayerPerceptronKokkos::evaluate_gradient_directional",
        Kokkos::TeamPolicy<>(batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int batch = team_member.league_rank();

            for (int l=0; l<shape.size()-1; ++l) {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, weights(l).extent(0)),
                    [=] (const int output) {
                        double z = 0.0;
                        double z_dot = 0.0;
                        for (int input=0; input<weights(l).extent(1); ++input) {
                            const double weight = weights(l)(output,input);
                            z += weight*values(l)(batch,input);
                            z_dot += weight*value_dots(l)(batch,input);
                        }
                        if (l == shape.size()-2) {
                            values(l+1)(batch,output) = z;
                            value_dots(l+1)(batch,output) = z_dot;
                        } else {
                            const double sigmoid = 1.0/(1.0+Kokkos::exp(-z));
                            const double activation_derivative = activation_scale
                                *(sigmoid + z*sigmoid*(1.0-sigmoid));
                            values(l+1)(batch,output) = activation_scale*z*sigmoid;
                            value_dots(l+1)(batch,output) =
                                activation_derivative*z_dot;
                        }
                    });
                team_member.team_barrier();
            }

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(
                    team_member, derivatives(shape.size()-2).extent(1)),
                [=] (const int input) {
                    derivatives(shape.size()-2)(batch,input) =
                        weights(shape.size()-2)(0,input);
                    derivative_dots(shape.size()-2)(batch,input) = 0.0;
                });
            team_member.team_barrier();

            for (int l=shape.size()-3; l>=0; --l) {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, weights(l).extent(0)),
                    [=] (const int output) {
                        double z = 0.0;
                        double z_dot = 0.0;
                        for (int input=0; input<weights(l).extent(1); ++input) {
                            const double weight = weights(l)(output,input);
                            z += weight*values(l)(batch,input);
                            z_dot += weight*value_dots(l)(batch,input);
                        }
                        const double sigmoid = 1.0/(1.0+Kokkos::exp(-z));
                        const double sigmoid_derivative = sigmoid*(1.0-sigmoid);
                        const double activation_derivative = activation_scale
                            *(sigmoid + z*sigmoid_derivative);
                        const double activation_second_derivative = activation_scale
                            *(2.0*sigmoid_derivative
                              +z*sigmoid_derivative*(1.0-2.0*sigmoid));
                        derivative_dots(l+1)(batch,output) =
                            derivative_dots(l+1)(batch,output)*activation_derivative
                            +derivatives(l+1)(batch,output)
                                *activation_second_derivative*z_dot;
                        derivatives(l+1)(batch,output) *= activation_derivative;
                    });
                team_member.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, weights(l).extent(1)),
                    [=] (const int input) {
                        double derivative = 0.0;
                        double derivative_dot = 0.0;
                        for (int output=0; output<weights(l).extent(0); ++output) {
                            derivative += weights(l)(output,input)
                                *derivatives(l+1)(batch,output);
                            derivative_dot += weights(l)(output,input)
                                *derivative_dots(l+1)(batch,output);
                        }
                        derivatives(l)(batch,input) = derivative;
                        derivative_dots(l)(batch,input) = derivative_dot;
                    });
                team_member.team_barrier();
            }
        });

    Kokkos::deep_copy(f,
        Kokkos::subview(node_values(shape.size()-1), Kokkos::ALL, 0));
    Kokkos::deep_copy(g, node_derivatives(0));
    Kokkos::deep_copy(g_dot, node_derivative_dots(0));
    Kokkos::fence();
}
