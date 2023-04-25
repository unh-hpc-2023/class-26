
#include "wave_equation.h"

#include <xtensor/xpad.hpp>

xt::xtensor<double, 1> derivative(const xt::xtensor<double, 1>& f, double dx)
{
  const int G = 1;
  int n = f.shape(0);

  auto f_g = xt::pad(f, G);
  f_g(G + -1) = f_g(G + n - 1);
  f_g(G + n) = f_g(G + 0);

  auto fprime = xt::zeros_like(f);
  for (int i = 0; i < n; i++) {
    fprime(i) = (f_g(G + i + 1) - f_g(G + i + -1)) / (2. * dx);
  }

  return fprime;
}
