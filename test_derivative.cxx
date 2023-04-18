
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

#include <fstream>
#include <iostream>

xt::xtensor<double, 1> derivative(const xt::xtensor<double, 1>& f, double dx)
{
  int N = f.shape(0);

  auto fprime = xt::zeros_like(f);
  fprime(0) = (f(1) - f(N - 1)) / (2. * dx);
  for (int i = 1; i < N - 1; i++) {
    fprime(i) = (f(i + 1) - f(i - 1)) / (2. * dx);
  }
  fprime(N - 1) = (f(0) - f(N - 2)) / (2. * dx);

  return fprime;
}

int main(int argc, char** argv)
{
  // create coordinates [0, 2pi)
  auto x = xt::linspace<double>(0., 2. * M_PI, 16, false);
  double dx = x(1) - x(0);
  // our original function f
  auto f = sin(x);

  // calculate derivative
  auto fprime = derivative(f, dx);

  std::ofstream out("f.csv");
  xt::dump_csv(out, xt::stack(xt::xtuple(x, f, fprime), 1));

  return 0;
}
