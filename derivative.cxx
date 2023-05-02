
#include "wave_equation.h"

#include <xtensor/xpad.hpp>

#include <mpi.h>

xt::xtensor<double, 1> derivative(const mpi_domain& domain,
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
