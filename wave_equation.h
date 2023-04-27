
#ifndef WAVE_EQUATION_H
#define WAVE_EQUATION_H

#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

#include <mpi.h>

class mpi_domain
{
public:
  mpi_domain(MPI_Comm comm, int N, double L) : comm_(comm), N_(N), L_(L)
  {
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);

    // number of grid points must be divisible by number of procs
    // (this could be relaxed, but needs to be handled properly)
    assert(N_ % size_ == 0);
    n_ = N_ / size_;
    dx_ = L_ / N_;
  }

  MPI_Comm comm() const { return comm_; }
  int rank() const { return rank_; }
  int size() const { return size_; }
  int N() const { return N_; }
  int n() const { return n_; }
  double dx() const { return dx_; }

private:
  MPI_Comm comm_;
  int rank_;
  int size_;
  int N_;     // global number of grid points
  int n_;     // local number of grid points (on this proc)
  double L_;  // domain length
  double dx_; // grid spacing
};

xt::xtensor<double, 1> derivative(const xt::xtensor<double, 1>& f, double dx);

#endif
