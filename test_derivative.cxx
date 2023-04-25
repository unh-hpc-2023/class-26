
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>

#include <fstream>
#include <iostream>

#include <mpi.h>

xt::xtensor<double, 1> derivative(const xt::xtensor<double, 1>& f, double dx)
{
  const int G = 1;
  int N = f.shape(0);

  auto f_g = xt::pad(f, G);
  f_g(G + -1) = f_g(G + N - 1);
  f_g(G + N) = f_g(G + 0);

  auto fprime = xt::zeros_like(f);
  for (int i = 0; i < N; i++) {
    fprime(i) = (f_g(i + G + 1) - f_g(i + G - 1)) / (2. * dx);
  }

  return fprime;
}

int main(int argc, char** argv)
{
  const int N = 16; // number of grid points

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // create coordinates [0, 2pi)
  double dx = 2. * M_PI / N;
  auto x = xt::arange<double>(0., 2. * M_PI, dx);

  // our original function f
  auto f = sin(x);

  // calculate derivative
  auto fprime = derivative(f, dx);

  std::ofstream out("f.csv");
  xt::dump_csv(out, xt::stack(xt::xtuple(x, f, fprime), 1));

  MPI_Finalize();
  return 0;
}
