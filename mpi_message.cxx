#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int test = 0;

  if (rank == 0) {
    test = 99;
    for (int r = 0; r < size; r++) {
      MPI_Send(&test, 1, MPI_INT, r, 123, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&test, 1, MPI_INT, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  printf("Hello %d/%d test = %d\n", rank, size, test);
  MPI_Finalize();
  return 0;
}
