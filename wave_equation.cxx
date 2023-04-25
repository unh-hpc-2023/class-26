
#include "wave_equation.h"

#include <xtensor/xcsv.hpp>

#include <mpi.h>

#include <cassert>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
  const int N = 16;            // number of grid points
  const int n_timesteps = 100; // total number of steps to run
  const int output_every = 1;  // how often to write output
  const double c = 1;          // wave speed

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

  // our initial condition
  xt::xtensor<double, 1> u = sin(x);

  double dt = dx;
  for (int n = 0; n < n_timesteps; n++) {
    if (n % output_every == 0) {
      std::ofstream out("u-" + std::to_string(n) + "-" + std::to_string(rank) +
                        ".csv");
      xt::dump_csv(out, xt::stack(xt::xtuple(x, u), 1));
    }

    // advance one timestep
    auto k1 = -c * derivative(u, dx);
    auto k2 = -c * derivative(u + .5 * dt * k1, dx);
    u += dt * k2;
  }

  MPI_Finalize();
  return 0;
}
