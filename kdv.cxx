
#include "wave_equation.h"

#include <xtensor/xcsv.hpp>
#include <xtensor/xpad.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

static double t_fill_ghosts;
static double t_compute_rhs;
static double t_output;

xt::xtensor<double, 1> rhs(const mpi_domain& domain,
                           const xt::xtensor<double, 1>& u)
{
  const int G = 2;
  int n = domain.n();
  assert(u.shape(0) == n);

  auto rhs = xt::zeros_like(u);
  auto u_g = xt::pad(u, G);

  t_fill_ghosts -= MPI_Wtime();
  domain.fill_ghosts(u_g);
  t_fill_ghosts += MPI_Wtime();

  t_compute_rhs -= MPI_Wtime();
  double dx3i = 1. / (2 * std::pow(domain.dx(), 3));
  double dxi = 1. / (2. * domain.dx());

  for (int i = 0; i < n; i++) {
    rhs(i) = (u_g(G + i + 2) - 2 * u_g(G + i + 1) + 2 * u_g(G + i - 1) -
              u_g(G + i - 2)) *
               dx3i -
             6 * u_g(G + i) * (u_g(G + i + 1) - u_g(G + i - 1)) * dxi;
  }
  t_compute_rhs += MPI_Wtime();

  return rhs;
}

void write_output(const mpi_domain& domain, const xt::xtensor<double, 1>& x,
                  const xt::xtensor<double, 1>& u, int t)
{
  std::ofstream out("u-" + std::to_string(t) + "-" +
                    std::to_string(domain.rank()) + ".csv");
  xt::dump_csv(out, xt::stack(xt::xtuple(x, u), 1));
}

void write_output_onefile(const mpi_domain& domain,
                          const xt::xtensor<double, 1>& x,
                          const xt::xtensor<double, 1>& u, int t)
{
  t_output -= MPI_Wtime();
  auto X = xt::empty<double>({domain.N()});
  auto U = xt::empty<double>({domain.N()});

  MPI_Gather(x.data(), x.size(), MPI_DOUBLE, X.data(), x.size(), MPI_DOUBLE, 0,
             domain.comm());
  MPI_Gather(u.data(), u.size(), MPI_DOUBLE, U.data(), u.size(), MPI_DOUBLE, 0,
             domain.comm());

  if (domain.rank() == 0) {
    std::ofstream out("u-" + std::to_string(t) + ".csv");
    xt::dump_csv(out, xt::stack(xt::xtuple(X, U), 1));
  }
  t_output += MPI_Wtime();
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

  double t_beg = MPI_Wtime();
  for (int n = 0; n < n_timesteps; n++) {
    if (output_every > 0 && n % output_every == 0) {
      write_output_onefile(domain, x, u, n);
      if (domain.rank() == 0) {
        std::cout << "writing output " << n << "\n";
      }
    }

    // advance one timestep
    auto k1 = rhs(domain, u);
    auto k2 = rhs(domain, u + .5 * dt * k1);
    auto k3 = rhs(domain, u + .5 * dt * k2);
    auto k4 = rhs(domain, u + dt * k3);
    u += (1. / 6.) * dt * (k1 + 2. * k2 + 2. * k3 + k4);
  }

  double t_end = MPI_Wtime();
  if (domain.rank() == 0) {
    std::cout << "entire time integration: " << t_end - t_beg << " secs\n";
    std::cout << "fill_ghosts            : " << t_fill_ghosts << " secs\n";
    std::cout << "compute_rhs            : " << t_compute_rhs << " secs\n";
    std::cout << "output                 : " << t_output << " secs\n";
  }

  MPI_Finalize();
  return 0;
}
