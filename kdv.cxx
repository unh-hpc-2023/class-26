
#include "wave_equation.h"

#include <xtensor/xcsv.hpp>
#include <xtensor/xpad.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

xt::xtensor<double, 1> rhs(const mpi_domain& domain,
                           const xt::xtensor<double, 1>& f)
{
  const int G = 1;
  int n = domain.n();
  assert(f.shape(0) == n);

  auto f_g = xt::pad(f, G);
  domain.fill_ghosts(f_g);

  auto fprime = xt::zeros_like(f);
  for (int i = 0; i < n; i++) {
    fprime(i) = (f_g(G + i + 1) - f_g(G + i + -1)) / (2. * domain.dx());
  }

  return fprime;
}

int main(int argc, char** argv)
{
  const int N = 16;            // number of grid points
  const double L = 2. * M_PI;  // total size of domain
  const int n_timesteps = 100; // total number of steps to run
  const int output_every = 1;  // how often to write output
  const double c = 1;          // wave speed

  MPI_Init(&argc, &argv);
  mpi_domain domain(MPI_COMM_WORLD, N, L);

  // create coordinates [0, 2pi)
  auto x = domain.coords();

  // our initial condition
  xt::xtensor<double, 1> u = sin(x);

  double dt = domain.dx();
  for (int n = 0; n < n_timesteps; n++) {
    if (n % output_every == 0) {
      std::ofstream out("u-" + std::to_string(n) + "-" +
                        std::to_string(domain.rank()) + ".csv");
      xt::dump_csv(out, xt::stack(xt::xtuple(x, u), 1));
    }

    // advance one timestep
    auto k1 = -c * rhs(domain, u);
    auto k2 = -c * rhs(domain, u + .5 * dt * k1);
    u += dt * k2;
  }

  MPI_Finalize();
  return 0;
}
