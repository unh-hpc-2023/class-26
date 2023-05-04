
#include "wave_equation.h"

#include <xtensor/xcsv.hpp>
#include <xtensor/xpad.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

xt::xtensor<double, 1> rhs(const mpi_domain& domain,
                           const xt::xtensor<double, 1>& u)
{
  const int G = 2;
  int n = domain.n();
  assert(u.shape(0) == n);

  auto u_g = xt::pad(u, G);
  domain.fill_ghosts(u_g);

  double dx3i = 1. / (2 * std::pow(domain.dx(), 3));
  double dxi = 1. / (2. * domain.dx());

  auto rhs = xt::zeros_like(u);
  for (int i = 0; i < n; i++) {
    rhs(i) = (u_g(G + i + 2) - 2 * u_g(G + i + 1) + 2 * u_g(G + i - 1) -
              u_g(G + i - 2)) *
               dx3i -
             6 * u_g(G + i) * (u_g(G + i + 1) - u_g(G + i - 1)) * dxi;
  }

  return rhs;
}

int main(int argc, char** argv)
{
  const int N = 128;              // number of grid points
  const double L = 2. * M_PI;     // total size of domain
  const int n_timesteps = 100000; // total number of steps to run
  const int output_every = 1000;  // how often to write output

  MPI_Init(&argc, &argv);
  mpi_domain domain(MPI_COMM_WORLD, N, L);

  // create coordinates [0, 2pi)
  auto x = domain.coords();

  // our initial condition
#if 0
  xt::xtensor<double, 1> u = sin(x);
#else
  double c = 3.;
  xt::xtensor<double, 1> u =
    -.5 * c / xt::pow(xt::cosh(.5 * std::sqrt(c) * (x - M_PI)), 2);
#endif

  // a very stringent CLF condition
  double dt = .1 * std::pow(domain.dx(), 3);
  for (int n = 0; n < n_timesteps; n++) {
    if (n % output_every == 0) {
      std::cout << "writing output " << n << "\n";
      std::ofstream out("u-" + std::to_string(n) + "-" +
                        std::to_string(domain.rank()) + ".csv");
      xt::dump_csv(out, xt::stack(xt::xtuple(x, u), 1));
    }

    // advance one timestep
    auto k1 = rhs(domain, u);
    auto k2 = rhs(domain, u + .5 * dt * k1);
    u += dt * k2;
  }

  MPI_Finalize();
  return 0;
}
