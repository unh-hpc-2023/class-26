#ifndef MPI_DOMAIN_H
#define MPI_DOMAIN_H

#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

#include <mpi.h>

#include <cassert>

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

  auto coords() const
  {
    return xt::arange<double>(rank() * n(), (rank() + 1) * n()) * dx();
  }

  void fill_ghosts(xt::xtensor<double, 1>& f_g) const
  {
    int n = this->n();
    const int G = (f_g.shape(0) - n) / 2;

    const int tag_left_to_right = 123;
    const int tag_right_to_left = 456;

    int rank_right = (rank() < size() - 1) ? rank() + 1 : 0;
    int rank_left = (rank() > 0) ? rank() - 1 : size() - 1;

    MPI_Request reqs[4];
    MPI_Irecv(&f_g(G - G), G, MPI_DOUBLE, rank_left, tag_left_to_right,
              MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&f_g(G + n), G, MPI_DOUBLE, rank_right, tag_right_to_left,
              MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend(&f_g(G + n - G), G, MPI_DOUBLE, rank_right, tag_left_to_right,
              MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&f_g(G + 0), G, MPI_DOUBLE, rank_left, tag_right_to_left,
              MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
  }

private:
  MPI_Comm comm_;
  int rank_;
  int size_;
  int N_; // global number of grid points
  int n_; // local number of grid points (on this proc)
  double L_;
  double dx_;
};

#endif
