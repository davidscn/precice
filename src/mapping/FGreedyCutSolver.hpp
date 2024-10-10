#pragma once
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <numeric>
#include "io/ExportVTU.hpp"
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include <fenv.h>
#include "mapping/config/MappingConfiguration.hpp"

namespace precice {
namespace mapping {


template <typename RADIAL_BASIS_FUNCTION_T>
class FGreedyCutSolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  FGreedyCutSolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
  */
  template <typename IndexContainer>
  FGreedyCutSolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
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
  precice::logging::Logger _log{"mapping::FGreedyCutSolver"};

  std::pair<int, double> select(const Eigen::VectorXd &residual) const;

  Eigen::MatrixXd buildEvaluationMatrix(const std::vector<int> &greedyIDs) const;
  void updateKernelVector(const mesh::Vertex &x, Eigen::VectorXd &kernelVector) const;
  void recalculateResidual(const Eigen::VectorXd &inputData, Eigen::VectorXd &r, Eigen::VectorXd &residual);

  /// max iterations
  size_t _maxIter;

  /// n_randon
  double _tolF;

  size_t _inSize  = 0;
  size_t _outSize = 0;
  Eigen::MatrixXd _kernelEval;
  std::vector<int> _greedyIDs;

  Eigen::MatrixXd _kernelMatrix;
  Eigen::MatrixXd _cut;

  const std::shared_ptr<precice::mesh::Mesh::VertexContainer> _inputVertices;
  const std::shared_ptr<precice::mesh::Mesh::VertexContainer> _outputVertices;
  RADIAL_BASIS_FUNCTION_T _basisFunction;
  size_t _basisSize;
  std::array<bool, 3> _activeAxis;
};

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::select(const Eigen::VectorXd &residual) const
{
  Eigen::Index maxIndex;
  double maxValue = residual.maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::buildEvaluationMatrix(const std::vector<int> &greedyIDs) const
{
  Eigen::MatrixXd matrixA(greedyIDs.size(), _outputVertices->size());

  for (size_t i = 0; i < greedyIDs.size(); i++) 
  {
    const auto &u = _inputVertices->at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < _outputVertices->size(); j++) 
    {
      const auto  &v = _outputVertices->at(j).rawCoords();
      const double squaredDifference = computeSquaredDifference(u, v, _activeAxis);
      matrixA(i, j) = _basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  return matrixA;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::recalculateResidual(const Eigen::VectorXd &inputData, Eigen::VectorXd &interpolationCoeffs, Eigen::VectorXd &residual) {
  int n = _greedyIDs.size();
  const auto &v = _inputVertices->at(_greedyIDs.at(n - 1)).rawCoords();
  for (size_t i = 0; i < _inSize; i++) 
  {
    const auto &u = _inputVertices->at(i).rawCoords();
    const double squaredDifference = computeSquaredDifference(u, v, _activeAxis);
    _kernelMatrix(i, n-1) = _basisFunction.evaluate(std::sqrt(squaredDifference));
  }
  const double a = (_cut.block(n - 1, 0, 1, n) * inputData(_greedyIDs)).value();
  interpolationCoeffs.head(n) += a * _cut.block(n - 1, 0, 1, n).transpose();
  residual = (inputData - _kernelMatrix.block(0, 0, _inSize, n) * interpolationCoeffs.head(n)).cwiseAbs();
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const mesh::Vertex &x, Eigen::VectorXd &kernelVector) const
{
  for (size_t j = 0; j < _greedyIDs.size(); j++)
  {
    const auto &y = _inputVertices->at(_greedyIDs.at(j)).rawCoords();
    kernelVector(j) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis))); 
  }
}


// ---- Initialization ---- //


template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::FGreedyCutSolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial, MappingConfiguration::GreedyParameter greedyParameter)
  : _inSize(inputMesh.vertices().size()), 
    _outSize(outputMesh.vertices().size()), 
    _inputVertices(std::make_shared<mesh::Mesh::VertexContainer>(std::move(inputMesh.vertices()))),
    _outputVertices(std::make_shared<mesh::Mesh::VertexContainer>(std::move(outputMesh.vertices()))),
    _basisFunction(basisFunction)
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(_kernelEval.size() == 0); 

  _tolF = greedyParameter.tolerance;
  _maxIter = greedyParameter.maxIterations;
  _basisSize = std::min(_inSize, _maxIter);
  _kernelMatrix = Eigen::MatrixXd::Zero(_inSize, _basisSize);
  _cut = Eigen::MatrixXd::Zero(_basisSize, _basisSize);
  _greedyIDs.reserve(_basisSize);

  _activeAxis = std::array<bool, 3>({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), _activeAxis.begin(), [](const auto ax) { return !ax; });
}


// ---- Evaluation ---- //


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial)
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial)
{
  Eigen::VectorXd residual = inputData;
  Eigen::VectorXd kernelVectorOldCenters = Eigen::VectorXd::Ones(_basisSize);
  Eigen::VectorXd basisVector = Eigen::VectorXd::Ones(_basisSize);
  Eigen::VectorXd interpolationCoeffs = Eigen::VectorXd::Zero(_basisSize);
  _greedyIDs.clear();

  const double kernelDiagonal = _basisFunction.evaluate(0);

  // Iterative selection of new points
  for (size_t n = 0; n < _basisSize; ++n) {

    auto [i, fMax] = select(residual);
    auto x = _inputVertices->at(i); 

    updateKernelVector(x, kernelVectorOldCenters);
    basisVector.head(n) = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * kernelVectorOldCenters.head(n);
    const double squareP = kernelDiagonal - basisVector.array().head(n).square().sum();

    if (fMax < _tolF || squareP <= 0) break;
    const double invP = 1.0 / std::sqrt(squareP);

    _cut.block(n, 0, 1, n).noalias() = -basisVector.block(0, 0, n, 1).transpose() * _cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _cut(n, n) = 1;
    _cut.block(n, 0, 1, n + 1) *= invP;

    _greedyIDs.push_back(i);
    recalculateResidual(inputData, interpolationCoeffs, residual);

    PRECICE_DEBUG("Iteration: {}, fMax = {}, P² = {}", n + 1, fMax, squareP);
  }
  const size_t n = _greedyIDs.size();
  Eigen::MatrixXd kernelEval = buildEvaluationMatrix(_greedyIDs);
  Eigen::VectorXd prediction = kernelEval.transpose() * interpolationCoeffs.head(n);

  return prediction;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _kernelEval   = Eigen::MatrixXd();
  _kernelMatrix = Eigen::MatrixXd();
  _cut          = Eigen::MatrixXd();
  _inSize  = 0;
  _outSize = 0;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedyCutSolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice
