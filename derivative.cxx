
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

  assert(size == 4);
  if (rank < 3) {
    int rank_right = rank + 1;
    MPI_Send(&f_g(G + n - 1), 1, MPI_DOUBLE, rank_right, 123, MPI_COMM_WORLD);
    MPI_Recv(&f_g(G + n), 1, MPI_DOUBLE, rank_right, 123, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  if (rank == 1) {
    MPI_Send(&f_g(G + 0), 1, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD);
    MPI_Recv(&f_g(G - 1), 1, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  if (rank == 2) {
    MPI_Send(&f_g(G + 0), 1, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
    MPI_Recv(&f_g(G - 1), 1, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  if (rank == 3) {
    MPI_Send(&f_g(G + 0), 1, MPI_DOUBLE, 2, 123, MPI_COMM_WORLD);
    MPI_Recv(&f_g(G - 1), 1, MPI_DOUBLE, 2, 123, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  if (rank == 0) {
    MPI_Send(&f_g(G + 0), 1, MPI_DOUBLE, 3, 456, MPI_COMM_WORLD);
    MPI_Recv(&f_g(G - 1), 1, MPI_DOUBLE, 3, 456, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else if (rank == 3) {
    MPI_Send(&f_g(G + n - 1), 1, MPI_DOUBLE, 0, 456, MPI_COMM_WORLD);
    MPI_Recv(&f_g(G + n), 1, MPI_DOUBLE, 0, 456, MPI_COMM_WORLD,
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
