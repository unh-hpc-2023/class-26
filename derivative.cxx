
#include "wave_equation.h"

#include <xtensor/xpad.hpp>

#include <mpi.h>

void fill_ghosts(xt::xtensor<double, 1>& f_g)
{
  const int G = 1;
  int n = f_g.shape(0) - 2 * G;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(size == 2);
  if (rank == 0) {
    MPI_Send(&f_g(G + n - 1), 1, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&f_g(G - 1), 1, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
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
