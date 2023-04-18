
#include <cmath>
#include <iostream>

#include "wtime.h"

double f(double x) { return std::sqrt(1. - x * x); }

int main(int argc, char** argv)
{
  const int N = 100000000;

  double t1 = Wtime();
  double sum = 0.;
  double dx = 1. / N;
  for (int i = 0; i < N; i++) {
    sum += .5 * dx * (f(i * dx) + f((i + 1) * dx));
  }
  double t2 = Wtime();
  std::cout << "time " << t2 - t1 << " secs\n";

  std::cout << "integral is approximately " << sum << "\n";

  return 0;
}
