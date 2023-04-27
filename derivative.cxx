
#include "wave_equation.h"

#include <xtensor/xpad.hpp>

void fill_ghosts(xt::xtensor<double, 1>& f_g)
{
  const int G = 1;
  int n = f_g.shape(0) - 2 * G;
  f_g(G + -1) = f_g(G + n - 1);
  f_g(G + n) = f_g(G + 0);
}

xt::xtensor<double, 1> derivative(const xt::xtensor<double, 1>& f, double dx)
{
  const int G = 1;
  int n = f.shape(0);

  auto f_g = xt::pad(f, G);
  fill_ghosts(f_g);

  auto fprime = xt::zeros_like(f);
  for (int i = 0; i < n; i++) {
    fprime(i) = (f_g(G + i + 1) - f_g(G + i + -1)) / (2. * dx);
  }

  return fprime;
}
