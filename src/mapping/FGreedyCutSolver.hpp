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

namespace precice {
namespace mapping {


template <typename RADIAL_BASIS_FUNCTION_T>
class FGreedySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  FGreedySolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
  */
  template <typename IndexContainer>
  FGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial);

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
  precice::logging::Logger _log{"mapping::FGreedySolver"};

  std::pair<int, double> select(const Eigen::VectorXd &residual) const;

  Eigen::MatrixXd buildEvaluationMatrix(const std::vector<int> &greedyIDs) const;
  void updateKernelVector(const std::vector<int> greedyIDs, Eigen::VectorXd &kernelVector, const mesh::Vertex &x) const;
  Eigen::VectorXd recalculateResidual(const std::vector<int> &greedyIDs, const Eigen::MatrixXd &cut, Eigen::MatrixXd &kernelMatrix, const Eigen::VectorXd &inputData, Eigen::VectorXd &r) const;

  /// max iterations
  const int _maxIter = 2000;

  /// n_randon
  const double _tolF = 1e-10;

  Eigen::Index    _inSize  = 0;
  Eigen::Index    _outSize = 0;
  Eigen::MatrixXd _kernelEval;

  const mesh::Mesh::VertexContainer _inputVertices;
  const mesh::Mesh::VertexContainer _outputVertices;
  RADIAL_BASIS_FUNCTION_T _basisFunction;
  int _basisSize;
  std::array<bool, 3> _activeAxis;
};

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedySolver<RADIAL_BASIS_FUNCTION_T>::select(const Eigen::VectorXd &residual) const
{
  Eigen::Index maxIndex;
  double maxValue = residual.maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedySolver<RADIAL_BASIS_FUNCTION_T>::buildEvaluationMatrix(const std::vector<int> &greedyIDs) const
{
  Eigen::MatrixXd matrixA(greedyIDs.size(), _outputVertices.size());

  for (size_t i = 0; i < greedyIDs.size(); i++) 
  {
    const auto &u = _inputVertices.at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < _outputVertices.size(); j++) 
    {
      const auto  &v = _outputVertices.at(j).rawCoords();
      const double squaredDifference = computeSquaredDifference(u, v, _activeAxis);
      matrixA(i, j) = _basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  return matrixA;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedySolver<RADIAL_BASIS_FUNCTION_T>::recalculateResidual(const std::vector<int> &greedyIDs, const Eigen::MatrixXd &cut, Eigen::MatrixXd &kernelMatrix, const Eigen::VectorXd &inputData, Eigen::VectorXd &c) const {
  int n = greedyIDs.size();
  const auto &v = _inputVertices.at(greedyIDs.at(n-1)).rawCoords();
  for (size_t i = 0; i < _inSize; i++) 
  {
    const auto &u = _inputVertices.at(i).rawCoords();
    const double squaredDifference = computeSquaredDifference(u, v, _activeAxis);
    kernelMatrix(i, n-1) = _basisFunction.evaluate(std::sqrt(squaredDifference));
  }
  const double a = (cut.block(n-1,0,1,n) * inputData(greedyIDs))(0,0);
  c.head(n) += a * cut.block(n-1,0,1,n).transpose();
  return (inputData - kernelMatrix.block(0,0,_inSize, n) * c.head(n)).cwiseAbs();
}



template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedySolver<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const std::vector<int> greedyIDs, Eigen::VectorXd &kernelVector, const mesh::Vertex &x) const
{
  for (size_t j = 0; j < greedyIDs.size(); j++)
  {
    const auto &y = _inputVertices.at(greedyIDs.at(j)).rawCoords();
    kernelVector(j) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis))); 
  }
}


// ---- Initialization ---- //


template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
FGreedySolver<RADIAL_BASIS_FUNCTION_T>::FGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial)
  : _inSize(inputMesh.vertices().size()), 
    _outSize(outputMesh.vertices().size()), 
    _inputVertices(inputMesh.vertices()),
    _outputVertices(outputMesh.vertices()),
    _basisFunction(basisFunction)
{
  precice::profiling::Event initEvent("FGreedy.initialize", profiling::Synchronize);

  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite());
  PRECICE_ASSERT(_kernelEval.size() == 0); 

  _basisSize = std::min(static_cast<int>(_inSize), _maxIter);

  _activeAxis = std::array<bool, 3>({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), _activeAxis.begin(), [](const auto ax) { return !ax; });

  initEvent.stop();
}


// ---- Evaluation ---- //


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  precice::profiling::Event solveEvent("FGreedy.solveConsistent", profiling::Synchronize);

  Eigen::VectorXd residual = inputData;
  Eigen::MatrixXd kernelMatrix = Eigen::MatrixXd::Zero(_inSize, _basisSize);
  Eigen::MatrixXd cut = Eigen::MatrixXd::Zero(_basisSize, _basisSize);
  std::vector<int> greedyIDs;
  greedyIDs.reserve(_basisSize);
  Eigen::VectorXd v = Eigen::VectorXd::Ones(_basisSize);
  Eigen::VectorXd v2 = Eigen::VectorXd::Ones(_basisSize);
  Eigen::VectorXd c = Eigen::VectorXd::Zero(_inSize);

  // Iterative selection of new points
  for (int n = 0; n < _basisSize; ++n) {

    auto [i, fMax] = select(residual);
    auto x = _inputVertices.at(i); 
    greedyIDs.push_back(i);

    if (fMax < _tolF) break;

    updateKernelVector(greedyIDs, v, x);
    v2.head(n) = cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * v.head(n);
    double p = std::sqrt(v2(n));  
    v.head(n) = v2.head(n) / p;

    cut.block(n, 0, 1, n).noalias() = -v.block(0, 0, n, 1).transpose() * cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    cut(n, n) = 1;
    cut.block(n, 0, 1, n + 1) /= v(n);

    residual = recalculateResidual(greedyIDs, cut, kernelMatrix, inputData, c);

    std::cout << "iteration = " << n << "\r";
  }

  Eigen::MatrixXd kernelEval = buildEvaluationMatrix(greedyIDs);
  
  Eigen::VectorXd y = inputData(greedyIDs);
  Eigen::VectorXd prediction = kernelEval.transpose() * c;

  solveEvent.stop();

  return prediction;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedySolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _kernelEval = Eigen::MatrixXd();
  _inSize     = 0;
  _outSize    = 0;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice
