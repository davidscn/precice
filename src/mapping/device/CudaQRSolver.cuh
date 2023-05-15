#ifdef PRECICE_WITH_CUDA
#pragma once

#include <ginkgo/ginkgo.hpp>

/**
 * Computes the QR decomposition using CUDA
*/
void computeQRDecompositionCuda(const int deviceId, const std::shared_ptr<gko::Executor> &exec, gko::matrix::Dense<> *A_Q, gko::matrix::Dense<> *R);

void solvewithQRDecompositionCuda(const int deviceId, gko::matrix::Dense<> *U, gko::matrix::Dense<> *x, gko::matrix::Dense<> *rhs, gko::matrix::Dense<> *matQ,  gko::matrix::Dense<> *in_vec);

#endif
