
#include "wave_equation.h"

#include <xtensor/xcsv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

#include <mpi.h>

int main(int argc, char** argv)
{
  const int N = 32; // number of grid points

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(N % size == 0); // # grid points must be divisible by # procs
  int n = N / size;

  // create coordinates [0, 2pi)
  double L = 2. * M_PI; // total size of domain
  double dx = L / N;
  auto x = xt::arange<double>(rank * n, (rank + 1) * n) * dx;

  // our original function f
  auto f = sin(x + 1);

  // calculate derivative
  auto fprime = derivative(f, dx);

  std::ofstream out("f-" + std::to_string(rank) + ".csv");
  xt::dump_csv(out, xt::stack(xt::xtuple(x, f, fprime), 1));

  MPI_Finalize();
  return 0;
}
