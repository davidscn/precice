#pragma once

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <fenv.h>
#include "mapping/RadialBasisFctBaseMapping.hpp"
#include <numeric>
#include "io/ExportVTU.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mapping/config/MappingConfiguration.hpp"
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include <iostream>
#include <fstream>

namespace precice {
namespace mapping {

template <typename RADIAL_BASIS_FUNCTION_T>
class FGreedyCholeskyMapping : public RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyParameter = MappingConfiguration::GreedyParameter;

public:

  FGreedyCholeskyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter);

  void computeMapping() final override;

  void mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) final override;

  void mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) final override;

  void clear() final override;

  std::string getName() const final override;

  //void tagMeshFirstRound() final override;
  //void tagMeshSecondRound() final override;

private:
  precice::logging::Logger _log{"mapping::RadialBasisFctMapping"};

  mesh::PtrMesh _inputMesh;
  mesh::PtrMesh _outputMesh;

  Eigen::MatrixXd _basisMatrix;
  Eigen::MatrixXd _decomposedV;
  Eigen::MatrixXd _kernelEval;

  std::vector<int>    _greedyIDs;
  std::array<bool, 3> _activeAxis;

  /// max iterations
  size_t _maxIter;
  /// n_randon
  double _tolF;

  size_t _inSize  = 0;
  size_t _outSize = 0;
  size_t _basisSize;

  std::pair<int, double> select(const Eigen::MatrixXd &residual) const;

  Eigen::MatrixXd buildEvaluationMatrix(const std::vector<int> &greedyIDs) const;
  void            updateKernelVector(const mesh::Vertex &x, Eigen::VectorXd &kernelVector) const;

};


template <typename RADIAL_BASIS_FUNCTION_T>
FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::FGreedyCholeskyMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, Mapping::InitialGuessRequirement::None)
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off"); // TODO: Add correct asserts
  _tolF    = greedyParameter.tolerance * greedyParameter.tolerance;
  _maxIter = greedyParameter.maxIterations;

  _activeAxis = std::array<bool, 3>({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), _activeAxis.begin(), [](const auto ax) { return !ax; });
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::select(const Eigen::MatrixXd &residual) const
{
  Eigen::Index maxIndex;
  double       maxValue = residual.colwise().squaredNorm().maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::buildEvaluationMatrix(const std::vector<int> &greedyIDs) const
{
  const mesh::Mesh::VertexContainer &inputVertices  = _inputMesh->vertices();
  const mesh::Mesh::VertexContainer &outputVertices = _outputMesh->vertices();
  Eigen::MatrixXd matrixA(greedyIDs.size(), outputVertices.size());

  for (size_t i = 0; i < greedyIDs.size(); i++) 
  {
    const auto &u = inputVertices.at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < outputVertices.size(); j++) 
    {
      const auto & v = outputVertices.at(j).rawCoords();
      const double d = computeSquaredDifference(u, v, _activeAxis);
      matrixA(i, j)  = _basisFunction.evaluate(std::sqrt(d));
    }
  }
  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const mesh::Vertex &x, Eigen::VectorXd &kernelVector) const
{
  const mesh::Mesh::VertexContainer &inputVertices = _inputMesh->vertices();
  for (size_t j = 0; j < inputVertices.size(); j++) {
    const auto &y   = inputVertices.at(j).rawCoords();
    kernelVector(j) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis)));
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_kernelEval.size() == 0);

  precice::profiling::Event e("map.f-greedy-cholesky.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    _inputMesh  = this->output();
    _outputMesh = this->input();
  } else {
    _inputMesh  = this->input();
    _outputMesh = this->output();
  }

  _inSize  = _inputMesh->vertices().size();
  _outSize = _outputMesh->vertices().size();

  _basisSize   = std::min(_inSize, _maxIter);
  _basisMatrix = Eigen::MatrixXd::Zero(_inSize, _basisSize);
  _decomposedV = Eigen::MatrixXd::Zero(_basisSize, _basisSize);
  _greedyIDs.reserve(_basisSize);

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  precice::profiling::Event e("map.f-greedy-cholesky.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  const Eigen::MatrixXd inputData = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _inSize);

  Eigen::VectorXd basisVector(_inSize);
  Eigen::MatrixXd residual = inputData;
  _greedyIDs.clear();

  // Iterative selection of new points
  for (size_t n = 0; n < _basisSize; ++n) {

    const auto [i, fMax] = select(residual);
    const auto x         = _inputMesh->vertices().at(i);

    updateKernelVector(x, basisVector);
    basisVector -= _basisMatrix.block(0, 0, _inSize, n) * _basisMatrix.block(i, 0, 1, n).transpose();

    if (fMax < _tolF || basisVector(i) <= 0)
      break;
    _greedyIDs.push_back(i);

    const double invP = 1.0 / std::sqrt(basisVector(i));
    basisVector *= invP;
    _basisMatrix.col(n) = basisVector;
    _decomposedV.row(n) = _basisMatrix.row(i); // TODO: necessary?

    const Eigen::VectorXd newtonCoefficient = residual.col(i) * invP;
    residual -= newtonCoefficient * basisVector.transpose();

    PRECICE_DEBUG("Iteration: {}, fMax = {}, P = {}\n", n + 1, fMax, basisVector(i));
  }
  const size_t             n = _greedyIDs.size();
  const Eigen::IndexedView y = inputData(Eigen::all, _greedyIDs);

  const Eigen::MatrixXd kernelEval    = buildEvaluationMatrix(_greedyIDs);
  Eigen::MatrixXd interpolationCoeffs = _decomposedV.block(0, 0, n, n).triangularView<Eigen::Lower>().solve(y.transpose());
  _decomposedV.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>().solveInPlace(interpolationCoeffs);

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = kernelEval.transpose() * interpolationCoeffs.col(d);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.f-greedy-cholesky.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (f-cholesky-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedyCholeskyMapping<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _kernelEval  = Eigen::MatrixXd();
  _decomposedV = Eigen::MatrixXd();
  _basisMatrix = Eigen::MatrixXd();
  _inSize      = 0;
  _outSize     = 0;
}

} // namespace mapping
} // namespace precice
