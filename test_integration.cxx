
#include <cassert>
#include <cmath>
#include <iostream>

#include <mpi.h>

double f(double x) { return std::sqrt(1. - x * x); }

int main(int argc, char** argv)
{
  const int N = 100000000;

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  assert(N % size == 0); // only support evenly divisible number of trapezoids
  int ib = rank * (N / size);
  int ie = (rank + 1) * (N / size);

  double t1 = MPI_Wtime();
  double sum = 0.;
  double dx = 1. / N;
  for (int i = ib; i < ie; i++) {
    sum += .5 * dx * (f(i * dx) + f((i + 1) * dx));
  }
  double t2 = MPI_Wtime();
  std::cout << "time " << t2 - t1 << " secs\n";

  std::cout << "integral is approximately " << sum << "\n";

  MPI_Finalize();
  return 0;
}
