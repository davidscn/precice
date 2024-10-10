#pragma once
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <fenv.h>
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
class FGreedyCholeskySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  FGreedyCholeskySolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
  */
  template <typename IndexContainer>
  FGreedyCholeskySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                        const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial, MappingConfiguration::GreedyParameter greedyParameter);

  /// Maps the given input data
  Eigen::VectorXd solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial);

  /// Maps the given input data
  Eigen::VectorXd solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial);

  // Clear all stored matrices
  void clear();

  // Returns the size of the input data
  Eigen::Index getInputSize() const;

  // Returns the size of the input data
  Eigen::Index getOutputSize() const;

private:
  precice::logging::Logger _log{"mapping::FGreedyCholeskySolver"};

  std::pair<int, double> select(const Eigen::VectorXd &residual) const;

  Eigen::MatrixXd buildEvaluationMatrix(const std::vector<int> &greedyIDs) const;
  void            updateKernelVector(const mesh::Vertex &x, Eigen::VectorXd &kernelVector) const;

  /// max iterations
  size_t _maxIter;

  /// n_randon
  double _tolF;

  size_t          _inSize  = 0;
  size_t          _outSize = 0;
  size_t          _basisSize;
  Eigen::MatrixXd _kernelEval;

  const std::shared_ptr<precice::mesh::Mesh::VertexContainer> _inputVertices;
  const std::shared_ptr<precice::mesh::Mesh::VertexContainer> _outputVertices;

  Eigen::MatrixXd         _basisMatrix;
  Eigen::MatrixXd         _decomposedV;
  std::vector<int>        _greedyIDs;
  RADIAL_BASIS_FUNCTION_T _basisFunction;
  std::array<bool, 3>     _activeAxis;
};

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::select(const Eigen::VectorXd &residual) const
{
  Eigen::Index maxIndex;
  double       maxValue = residual.cwiseAbs().maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::buildEvaluationMatrix(const std::vector<int> &greedyIDs) const
{
  Eigen::MatrixXd matrixA(greedyIDs.size(), _outputVertices->size());

  for (size_t i = 0; i < greedyIDs.size(); i++) {
    const auto &u = _inputVertices->at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < _outputVertices->size(); j++) {
      const auto & v = _outputVertices->at(j).rawCoords();
      const double d = computeSquaredDifference(u, v, _activeAxis);
      matrixA(i, j)  = _basisFunction.evaluate(std::sqrt(d));
    }
  }

  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const mesh::Vertex &x, Eigen::VectorXd &kernelVector) const
{
  for (size_t j = 0; j < _inputVertices->size(); j++) {
    const auto &y   = _inputVertices->at(j).rawCoords();
    kernelVector(j) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis)));
  }
}

// ---- Initialization ---- //

template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::FGreedyCholeskySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial, MappingConfiguration::GreedyParameter greedyParameter)
    : _inSize(inputMesh.vertices().size()),
      _outSize(outputMesh.vertices().size()),
      _inputVertices(std::make_shared<mesh::Mesh::VertexContainer>(std::move(inputMesh.vertices()))),
      _outputVertices(std::make_shared<mesh::Mesh::VertexContainer>(std::move(outputMesh.vertices()))),
      _basisFunction(basisFunction)
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(_kernelEval.size() == 0);

  _tolF        = greedyParameter.tolerance;
  _maxIter     = greedyParameter.maxIterations;
  _basisSize   = std::min(_inSize, _maxIter);
  _basisMatrix = Eigen::MatrixXd::Zero(_inSize, _basisSize);
  _decomposedV = Eigen::MatrixXd::Zero(_basisSize, _basisSize);
  _greedyIDs.reserve(_basisSize);

  _activeAxis = std::array<bool, 3>({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), _activeAxis.begin(), [](const auto ax) { return !ax; });
}

// ---- Evaluation ---- //

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial)
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial)
{
  Eigen::VectorXd basisVector(_inSize);
  Eigen::VectorXd residual = inputData;
  _greedyIDs.clear();

  // Iterative selection of new points
  for (size_t n = 0; n < _basisSize; ++n) {

    auto [i, fMax] = select(residual);
    auto x         = _inputVertices->at(i);

    updateKernelVector(x, basisVector);
    basisVector -= _basisMatrix.block(0, 0, _inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();

    if (fMax < _tolF || basisVector(i) <= 0)
      break;
    _greedyIDs.push_back(i);

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    _basisMatrix.col(n) = basisVector;
    _decomposedV.row(n) = _basisMatrix.row(i); // TODO: necessary?

    const double newtonCoefficient = residual(i) * invP;
    residual -= newtonCoefficient * basisVector;

    PRECICE_DEBUG("Iteration: {}, fMax = {}, P = {}", n + 1, fMax, basisVector(i));
  }

  size_t             n = _greedyIDs.size();
  Eigen::IndexedView y = inputData(_greedyIDs);

  Eigen::MatrixXd kernelEval = buildEvaluationMatrix(_greedyIDs);

  Eigen::VectorXd interpolationCoeffs = _decomposedV.block(0, 0, n, n).triangularView<Eigen::Lower>().solve(y);
  _decomposedV.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>().solveInPlace(interpolationCoeffs);
  Eigen::VectorXd prediction = kernelEval.transpose() * interpolationCoeffs;

  return prediction;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _kernelEval  = Eigen::MatrixXd();
  _decomposedV = Eigen::MatrixXd();
  _basisMatrix = Eigen::MatrixXd();
  _inSize      = 0;
  _outSize     = 0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedyCholeskySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice
