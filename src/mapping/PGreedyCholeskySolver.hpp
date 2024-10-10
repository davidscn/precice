#pragma once
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include "io/ExportVTU.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mapping/config/MappingConfiguration.hpp"
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"

namespace precice {
namespace mapping {

template <typename RADIAL_BASIS_FUNCTION_T>
class PGreedyCholeskySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  PGreedyCholeskySolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
  */
  template <typename IndexContainer>
  PGreedyCholeskySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                        const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial, MappingConfiguration::GreedyParameter greedyParameter);

  /// Maps the given input data
  Eigen::VectorXd solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const;

  /// Maps the given input data
  Eigen::VectorXd solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const;

  // Clear all stored matrices
  void clear();

  // Returns the size of the input data
  Eigen::Index getInputSize() const;

  // Returns the size of the input data
  Eigen::Index getOutputSize() const;

private:
  precice::logging::Logger _log{"mapping::PGreedyCholeskySolver"};

  std::pair<int, double> select(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction);
  Eigen::MatrixXd        buildEvaluationMatrix(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &outputMesh, const mesh::Mesh &inputMesh, const std::array<bool, 3> &activeAxis);
  void                   updateKernelVector(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const std::array<bool, 3> &activeAxis, const mesh::Vertex &x, Eigen::VectorXd &kernelVector);

  size_t _inSize  = 0;
  size_t _outSize = 0;

  /// Selected mesh vertices used for interpolation
  std::vector<int> _greedyIDs;

  /// Full newton basis matrix
  Eigen::MatrixXd _basisMatrix;
  /// Reordered relevant part of the newton basis matrix: The cholesky decomposition of the kernel matrix
  Eigen::MatrixXd _decomposedV;
  /// Kernel evaluations on the in- and output mesh
  Eigen::MatrixXd _kernelEval;
  /// Power function evaluations for each input vertex: : iteratively updated
  Eigen::VectorXd _powerFunction;
};

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::select(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction)
{
  Eigen::Index maxIndex;
  double       maxValue = _powerFunction.maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::buildEvaluationMatrix(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &outputMesh, const mesh::Mesh &inputMesh, const std::array<bool, 3> &activeAxis)
{
  const mesh::Mesh::VertexContainer &inputVertices  = inputMesh.vertices();
  const mesh::Mesh::VertexContainer &outputVertices = outputMesh.vertices();

  Eigen::MatrixXd matrixA(_greedyIDs.size(), outputVertices.size());

  for (size_t i = 0; i < _greedyIDs.size(); i++) {
    const auto &u = inputVertices.at(_greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < outputVertices.size(); j++) {
      const auto & v = outputVertices.at(j).rawCoords();
      const double d = computeSquaredDifference(u, v, activeAxis);
      matrixA(i, j)  = basisFunction.evaluate(std::sqrt(d));
    }
  }

  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const std::array<bool, 3> &activeAxis, const mesh::Vertex &x, Eigen::VectorXd &kernelVector)
{
  const mesh::Mesh::VertexContainer &vertices = inputMesh.vertices();
  for (size_t j = 0; j < vertices.size(); j++) {
    const auto &y   = vertices.at(j).rawCoords();
    kernelVector(j) = basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, activeAxis)));
  }
}

// ---- Initialization ---- //

template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::PGreedyCholeskySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial,
                                                                      MappingConfiguration::GreedyParameter greedyParameter)
    : _inSize(inputMesh.vertices().size()), _outSize(outputMesh.vertices().size())
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_kernelEval.size() == 0);

  const int basisSize = std::min(_inSize, greedyParameter.maxIterations); // maximal number of used basis functions
  _powerFunction      = Eigen::VectorXd(_inSize);
  _powerFunction.fill(basisFunction.evaluate(0));
  _basisMatrix = Eigen::MatrixXd::Zero(_inSize, basisSize);
  _decomposedV = Eigen::MatrixXd::Zero(basisSize, basisSize);
  _greedyIDs.reserve(basisSize);

  Eigen::VectorXd basisVector(_inSize);

  // Convert dead axis vector into an active axis array so that we can handle the reduction more easily
  std::array<bool, 3> activeAxis({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), activeAxis.begin(), [](const auto ax) { return !ax; });

  // Iterative selection of new points
  for (int n = 0; n < basisSize; ++n) {

    auto [i, pMax] = select(inputMesh, basisFunction);
    auto x         = inputMesh.vertices().at(i);

    if (pMax < greedyParameter.tolerance)
      break;
    _greedyIDs.push_back(i);

    updateKernelVector(basisFunction, inputMesh, activeAxis, x, basisVector);
    basisVector -= _basisMatrix.block(0, 0, _inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();
    const double invP = 1.0 / std::sqrt(pMax);
    basisVector *= invP;

    _powerFunction -= (Eigen::VectorXd) basisVector.array().square();
    _basisMatrix.col(n) = basisVector;
    _decomposedV.row(n) = _basisMatrix.row(i); // TODO: necessary?

    PRECICE_DEBUG("Iteration: {}, pMax = {}", n + 1, pMax);
  }

  _kernelEval = buildEvaluationMatrix(basisFunction, outputMesh, inputMesh, activeAxis);
}

// ---- Evaluation ---- //

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  size_t             n = _greedyIDs.size();
  Eigen::IndexedView y = inputData(_greedyIDs);

  Eigen::VectorXd interpolationCoeffs = _decomposedV.block(0, 0, n, n).triangularView<Eigen::Lower>().solve(y);
  _decomposedV.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>().solveInPlace(interpolationCoeffs);
  Eigen::VectorXd prediction = _kernelEval.transpose() * interpolationCoeffs;

  return prediction;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _greedyIDs.clear();
  _kernelEval    = Eigen::MatrixXd();
  _powerFunction = Eigen::VectorXd();
  _basisMatrix   = Eigen::MatrixXd();
  _decomposedV   = Eigen::MatrixXd();
  _inSize        = 0;
  _outSize       = 0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice
