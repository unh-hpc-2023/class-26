
#include "wave_equation.h"

#include <xtensor/xcsv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
  const int N = 32; // number of grid points

  MPI_Init(&argc, &argv);
  mpi_domain domain(MPI_COMM_WORLD);

  assert(N % domain.size() == 0); // # grid points must be divisible by # procs
  int n = N / domain.size();

  // create coordinates [0, 2pi)
  double L = 2. * M_PI; // total size of domain
  double dx = L / N;
  auto x = xt::arange<double>(domain.rank() * n, (domain.rank() + 1) * n) * dx;

  // our original function f
  auto f = sin(x + 1);

  // calculate derivative
  auto fprime = derivative(f, dx);

  std::ofstream out("f-" + std::to_string(domain.rank()) + ".csv");
  xt::dump_csv(out, xt::stack(xt::xtuple(x, f, fprime), 1));

  MPI_Finalize();
  return 0;
}
