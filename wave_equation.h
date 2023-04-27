
#ifndef WAVE_EQUATION_H
#define WAVE_EQUATION_H

#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

#include <mpi.h>

class mpi_domain
{
public:
  mpi_domain(MPI_Comm comm) : comm_(comm)
  {
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
  }

  MPI_Comm comm() const { return comm_; }
  int rank() const { return rank_; }
  int size() const { return size_; }

private:
  MPI_Comm comm_;
  int rank_;
  int size_;
};

xt::xtensor<double, 1> derivative(const xt::xtensor<double, 1>& f, double dx);

#endif
