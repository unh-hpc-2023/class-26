
#include "mpi_domain.h"

#include <xtensor/xcsv.hpp>
#include <xtensor/xpad.hpp>

#include <fstream>
#include <iostream>

template <typename T>
auto sqr(T x)
{
  return x * x;
}

// ----------------------------------------------------------------------
// calculate the r.h.s of the heat equation
//
// u_t = kappa u_xx

inline xt::xtensor<double, 1> calc_rhs(const mpi_domain& domain,
                                       const xt::xtensor<double, 1>& f,
                                       double kappa)
{
  const int G = 1;

  auto f_g = xt::pad(f, G);
  domain.fill_ghosts(f_g);

  // Python equivalent: return kappa * (f_g[2:] - 2 * f_g[1;-1] + f_g[:-2]) /
  // (dx^2);
  double dx = domain.dx();
  return kappa *
         (xt::view(f_g, xt::range(2, _)) -
          2. * xt::view(f_g, xt::range(1, -1)) +
          xt::view(f_g, xt::range(_, -2))) /
         (dx * dx);
}

void write_output_onefile(const mpi_domain& domain,
                          const xt::xtensor<double, 1>& x,
                          const xt::xtensor<double, 1>& u, int t)
{
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
}

int main(int argc, char** argv)
{
  const int N = 100;
  const double max_time = 50.;
  const int output_every = 10;
  const double kappa = .01;

  MPI_Init(&argc, &argv);
  mpi_domain domain(MPI_COMM_WORLD, N, 2. * M_PI);

  double dt = .5 * sqr(domain.dx()) / kappa; // pick dt to satisfy CFL condition

  // create coordinates
  auto x = domain.coords();

  auto f = xt::eval(exp(-sqr(x - M_PI) / sqr(.5)));

  int n_timesteps = std::round(max_time / dt);
  for (int n = 0; n < n_timesteps; n++) {
    // write out current solution every so many steps
    if (output_every > 0 && n % output_every == 0) {
      write_output_onefile(domain, x, f, n);
    }

    // A simple forward Euler step x^{n+1} = x^{n} + dt * rhs(x^n)
    // works fine for integrating this equation:

    auto rhs = calc_rhs(domain, f, kappa);
    f += dt * rhs;
  }

  MPI_Finalize();
  return 0;
}
