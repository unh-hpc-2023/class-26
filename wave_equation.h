
#ifndef WAVE_EQUATION_H
#define WAVE_EQUATION_H

#include "mpi_domain.h"

xt::xtensor<double, 1> derivative(const mpi_domain& domain,
                                  const xt::xtensor<double, 1>& f);

#endif
