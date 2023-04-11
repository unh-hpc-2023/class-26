
#include <iostream>
#include <omp.h>

int main(int argc, char** argv)
{
  int test = 0;
#pragma omp parallel shared(test)
  {
    int thread_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    if (thread_id == 0) {
      test = 99;
    }
#pragma omp barrier
    printf("Hello %d/%d test = %d\n", thread_id, n_threads, test);
  }
}
