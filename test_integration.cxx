
#include <cmath>
#include <iostream>

#include <mpi.h>

double f(double x) { return std::sqrt(1. - x * x); }

int main(int argc, char** argv)
{
  const int N = 100000000;

  MPI_Init(&argc, &argv);
  double t1 = MPI_Wtime();
  double sum = 0.;
  double dx = 1. / N;
  for (int i = 0; i < N; i++) {
    sum += .5 * dx * (f(i * dx) + f((i + 1) * dx));
  }
  double t2 = MPI_Wtime();
  std::cout << "time " << t2 - t1 << " secs\n";

  std::cout << "integral is approximately " << sum << "\n";

  MPI_Finalize();
  return 0;
}
