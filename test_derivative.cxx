
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
  // create coordinates [0, 2pi)
  auto x = xt::linspace<double>(0., 2. * M_PI, 16, false);
  // our original function f
  auto f = sin(x);

  std::ofstream out("f.csv");
  xt::dump_csv(out, xt::stack(xt::xtuple(x, f), 1));

  return 0;
}
