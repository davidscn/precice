#ifdef PRECICE_WITH_CUDA
#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <ginkgo/ginkgo.hpp>
#include "mapping/QRSolver.hpp"

class CudaQRSolver : public QRSolver {
public:
  CudaQRSolver(const int deviceId = 0);
  void computeQR(const std::shared_ptr<gko::Executor> &exec, QRSolver::GinkgoMatrix *A_Q, QRSolver::GinkgoMatrix *R) final override;
  ~CudaQRSolver();

private:
  // Handles for low-level CUDA libraries
  cusolverDnHandle_t solverHandle;
  cusolverStatus_t   cusolverStatus = CUSOLVER_STATUS_SUCCESS;
  cudaError_t        cudaErrorCode  = cudaSuccess;
};

#endif