#ifdef PRECICE_WITH_CUDA
#pragma once

#include <ginkgo/ginkgo.hpp>

/**
 * Computes the QR decomposition using CUDA
*/
void computeQRDecompositionCuda(const int deviceId, const std::shared_ptr<gko::Executor> &exec, gko::matrix::Dense<> *A_Q, gko::matrix::Dense<> *R);

void solvewithQRDecompositionCuda(const int deviceId, gko::matrix::Dense<> *U, gko::matrix::Dense<> *rhs, bool lower);

#endif
