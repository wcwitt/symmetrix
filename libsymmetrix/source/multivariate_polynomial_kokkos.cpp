#include "KokkosBlas1_dot.hpp"

#include "multivariate_polynomial.hpp"
#include "tools_kokkos.hpp"

#include "multivariate_polynomial_kokkos.hpp"


MultivariatePolynomialKokkos::MultivariatePolynomialKokkos(
    int num_variables, 
    std::vector<double> coefficients,
    std::vector<std::vector<int>> monomials)
{
    // set num_variables, coefficients, and monomials
    this->num_variables = num_variables;
    set_kokkos_view(this->coefficients, coefficients);
    int max_monomial_size = 0;
    for (auto m : monomials)
        if (m.size() > max_monomial_size)
            max_monomial_size = m.size();
    Kokkos::realloc(this->monomials, monomials.size(), max_monomial_size);
    Kokkos::deep_copy(this->monomials, -1);
    auto h_monomials = Kokkos::create_mirror_view(this->monomials);
    for (int i=0; i<monomials.size(); ++i) {
        for (int j=0; j<monomials[i].size(); ++j) {
            h_monomials(i,j) = monomials[i][j];
        }
    }
    Kokkos::deep_copy(this->monomials, h_monomials);

    // create non-kokkos object, then extract graph information
    MultivariatePolynomial mp(num_variables, coefficients, monomials);
    num_auxiliary_nodes = mp.num_auxiliary_nodes;
    nodes = Kokkos::View<int**,Kokkos::LayoutRight>("nodes", mp.nodes.size(), max_monomial_size);
    Kokkos::deep_copy(nodes, -1);
    auto h_nodes = Kokkos::create_mirror_view(nodes);
    for (int i=0; i<mp.nodes.size(); ++i)
        for (int j=0; j<mp.nodes[i].size(); ++j)
            h_nodes(i,j) = mp.nodes[i][j];
    Kokkos::deep_copy(nodes, h_nodes);
    edges = Kokkos::View<int**,Kokkos::LayoutRight>("edges", mp.edges.size(), 2);
    auto h_edges = Kokkos::create_mirror_view(edges);
    for (int i=0; i<mp.edges.size(); ++i)
        for (int j=0; j<2; ++j)
            h_edges(i,j) = mp.edges[i][j];
    Kokkos::deep_copy(edges, h_edges);
    set_kokkos_view(node_coefficients, mp.node_coefficients);
    set_kokkos_view(node_values, mp.node_values);
    set_kokkos_view(node_adjoints, mp.node_adjoints);
}

double MultivariatePolynomialKokkos::evaluate(
    const Kokkos::View<const double*>& x)
{
    initialize_forward_pass(x);
    forward_pass();
    return KokkosBlas::dot(node_coefficients, node_values);
}

double MultivariatePolynomialKokkos::evaluate_simple(
    const Kokkos::View<const double*>& x)
{
    double f;
    Kokkos::parallel_reduce(
        "MultivariatePolynomialKokkos::evaluate_simple",
        coefficients.size(),
        KOKKOS_CLASS_LAMBDA (const int i, double& local_result) {
            double monomial = x(monomials(i,0));
            for (int j=1; j<monomials.extent(1); ++j) {
                if (monomials(i,j) == -1)
                    break;
                monomial *= x(monomials(i,j));
            }
            local_result += coefficients(i) * monomial;
        },
        f);
    return f;
}

double MultivariatePolynomialKokkos::evaluate_gradient(
    const Kokkos::View<const double*>& x,
    Kokkos::View<double*>& g)
{
    initialize_forward_pass(x);
    forward_pass();
    auto f = KokkosBlas::dot(node_coefficients, node_values);
    initialize_backward_pass();
    backward_pass();
    extract_gradient_from_graph(g);
    return f;
}

double MultivariatePolynomialKokkos::evaluate_gradient_simple(
    const Kokkos::View<const double*>& x,
    Kokkos::View<double*>& g)
{
    double f;
    Kokkos::parallel_reduce(
        "MultivariatePolynomialKokkos::evaluate_gradient_simple",
        coefficients.size(),
        KOKKOS_CLASS_LAMBDA (const int i, double& local_result) {
            double monomial = x(monomials(i,0));
            for (int j=1; j<monomials.extent(1); ++j) {
                if (monomials(i,j) == -1)
                    break;
                monomial *= x(monomials(i,j));
            }
            local_result += coefficients(i) * monomial;
        },
        f);
    Kokkos::deep_copy(g, 0.0);
    Kokkos::parallel_for(
        "MultivariatePolynomialKokkos::evaluate_gradient_simple",
        coefficients.size(),
        KOKKOS_CLASS_LAMBDA (const int i) {
            for (int j=0; j<monomials.extent(1); ++j) {
                if (monomials(i,j) == -1) break;
                double monomial_deriv = 1.0;
                for (int k=0; k<monomials.extent(1); ++k) {
                    if (monomials(i,k) == -1) break;
                    if (j == k) continue;
                    monomial_deriv *= x(monomials(i,k));
                }
                g(monomials(i,j)) += coefficients(i) * monomial_deriv;
            }
        });
    return f;
}

void MultivariatePolynomialKokkos::initialize_forward_pass(
    const Kokkos::View<const double*>& x)
{
    Kokkos::parallel_for(
        "MultivariatePolynomialKokkos::initialize_forward_pass",
        num_variables,
        KOKKOS_CLASS_LAMBDA (const int i) {
            node_values(i) = x(i);
        });
}

void MultivariatePolynomialKokkos::forward_pass()
{
    // TODO: parallelize
    for (int i=0; i<edges.extent(0); ++i) {
        const int i0 = edges(i,0);
        const int i1 = edges(i,1);
        node_values(num_variables+i) = node_values(i0)*node_values(i1);
    }
}

void MultivariatePolynomialKokkos::initialize_backward_pass()
{
    Kokkos::parallel_for(
        "MultivariatePolynomialKokkos::initialize_backward_pass",
        node_coefficients.size(),
        KOKKOS_CLASS_LAMBDA (const int i) {
            node_adjoints(i) = node_coefficients(i);
        });
}

void MultivariatePolynomialKokkos::backward_pass()
{
    for (int i=edges.extent(0)-1; i>=0; --i) {
        const int i0 = edges(i,0);
        const int i1 = edges(i,1);
        node_adjoints(i0) += node_adjoints(num_variables+i)*node_values(i1);
        node_adjoints(i1) += node_adjoints(num_variables+i)*node_values(i0);
    }
}

void MultivariatePolynomialKokkos::extract_gradient_from_graph(
    Kokkos::View<double*>& g)
{
    Kokkos::deep_copy(g, 0.0);
    Kokkos::parallel_for(
        "MultivariatePolynomialKokkos::extract_gradient_from_graph",
        num_variables,
        KOKKOS_CLASS_LAMBDA (const int i) {
            g(i) = node_adjoints(i);
        });
}
