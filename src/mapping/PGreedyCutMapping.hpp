#pragma once
#include <Eigen/Dense>
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
class PGreedyCutMapping : public RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T> {

  using RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>::_basisFunction;
  using GreedyParameter = MappingConfiguration::GreedyParameter;
  using VertexContainer = mesh::Mesh::VertexContainer;

public:

  PGreedyCutMapping(
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

private:
  precice::logging::Logger _log{"mapping::RadialBasisFctMapping"};

  mesh::PtrMesh _inputMesh;
  mesh::PtrMesh _outputMesh;

  Polynomial _polynomial;

  Eigen::MatrixXd _kernelEval;
  Eigen::MatrixXd _kernelMatrix;
  Eigen::MatrixXd _cut;
  Eigen::VectorXd _powerFunction;

  std::vector<int>    _greedyIDs;
  std::array<bool, 3> _activeAxis;

  /// max iterations
  size_t _maxIter;
  /// n_randon
  double _tolP;

  size_t _inSize  = 0;
  size_t _outSize = 0;
  size_t _basisSize;

  std::pair<int, double> select() const;

  Eigen::MatrixXd buildEvaluationMatrix(const std::vector<int> &greedyIDs) const;
  void updatePowerFunction(const mesh::Vertex &x, const std::vector<int> &greedyIDs);
  void updateKernelVector(const mesh::Vertex &x, std::vector<int> &greedyIDs, Eigen::VectorXd &kernelVector) const;

};


template <typename RADIAL_BASIS_FUNCTION_T>
PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::PGreedyCutMapping(
    Mapping::Constraint     constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    GreedyParameter         greedyParameter)
    : RadialBasisFctBaseMapping<RADIAL_BASIS_FUNCTION_T>(constraint, dimensions, function, deadAxis, Mapping::InitialGuessRequirement::None)
{
  PRECICE_ASSERT(polynomial != Polynomial::ON, "Poly off");
  _tolP    = greedyParameter.tolerance;
  _maxIter = greedyParameter.maxIterations;
  _polynomial = polynomial;

  _activeAxis = std::array<bool, 3>({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), _activeAxis.begin(), [](const auto ax) { return !ax; });
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::select() const {
  Eigen::Index maxIndex;
  double       maxValue = _powerFunction.maxCoeff(&maxIndex);
  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::MatrixXd PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::buildEvaluationMatrix(const std::vector<int> &greedyIDs) const {

  const VertexContainer &inputVertices  = _inputMesh->vertices();
  const VertexContainer &outputVertices = _outputMesh->vertices();
  Eigen::MatrixXd matrixA(greedyIDs.size(), outputVertices.size());

  for (size_t i = 0; i < greedyIDs.size(); i++) {
    const auto &u = inputVertices.at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < outputVertices.size(); j++) {
      const auto & v = outputVertices.at(j).rawCoords();
      const double d = computeSquaredDifference(u, v, _activeAxis);
      matrixA(i, j)  = _basisFunction.evaluate(std::sqrt(d));
    }
  }
  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::updateKernelVector(const mesh::Vertex &x, std::vector<int> &greedyIDs, Eigen::VectorXd &kernelVector) const {

  const mesh::Mesh::VertexContainer &inputVertices = _inputMesh->vertices();
  for (size_t j = 0; j < greedyIDs.size(); j++) {
    const auto &y   = inputVertices.at(greedyIDs.at(j)).rawCoords();
    kernelVector(j) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(x.rawCoords(), y, _activeAxis)));
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::updatePowerFunction(const mesh::Vertex &x, const std::vector<int> &greedyIDs) {

  const size_t n = greedyIDs.size() - 1;
  for (size_t j = 0; j < _inSize; j++) {
    const auto &y       = _inputMesh->vertices().at(j).rawCoords();
    _kernelMatrix(j, n) = _basisFunction.evaluate(std::sqrt(computeSquaredDifference(y, x.rawCoords(), _activeAxis)));
  }
  _powerFunction -= (Eigen::VectorXd)(_kernelMatrix.block(0, 0, _inSize, n + 1) * _cut.block(n, 0, 1, n + 1).transpose()).array().square();
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping() {
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_kernelEval.size() == 0);

  precice::profiling::Event e("map.P-greedy-cut.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    _inputMesh  = this->output();
    _outputMesh = this->input();
  } else {
    _inputMesh  = this->input();
    _outputMesh = this->output();
  }
  _inSize    = _inputMesh->vertices().size();
  _outSize   = _outputMesh->vertices().size();
  _basisSize = std::min(_inSize, _maxIter);

  _cut           = Eigen::MatrixXd::Zero(_basisSize, _basisSize);
  _powerFunction = Eigen::VectorXd(_inSize);
  _kernelMatrix  = Eigen::MatrixXd::Zero(_inSize, _basisSize);

  _powerFunction.fill(_basisFunction.evaluate(0));
  _greedyIDs.reserve(_basisSize);

  Eigen::VectorXd kernelVector = Eigen::VectorXd::Ones(_basisSize);
  Eigen::VectorXd basisVector  = Eigen::VectorXd::Ones(_basisSize);

  // Iterative selection of new points
  for (int n = 0; n < _basisSize; ++n) {

    auto [i, pMax] = select();
    auto x         = _inputMesh->vertices().at(i);

    if (pMax < _tolP)
      break;
    const double invP = 1.0 / std::sqrt(pMax);

    updateKernelVector(x, _greedyIDs, kernelVector);
    basisVector.head(n) = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * kernelVector.head(n);

    _cut.block(n, 0, 1, n).noalias() = -basisVector.block(0, 0, n, 1).transpose() * _cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _cut(n, n)                       = 1;
    _cut.block(n, 0, 1, n + 1) *= invP;

    _greedyIDs.push_back(i);
    updatePowerFunction(x, _greedyIDs);

    PRECICE_DEBUG("Iteration: {}, pMax = {}", n + 1, pMax);
  }
  _kernelEval = buildEvaluationMatrix(_greedyIDs);

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData) {
  
  precice::profiling::Event e("map.P-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);

  const Eigen::VectorXd &linearisedVectors = inData.values;
  const Eigen::MatrixXd inputData = Eigen::Map<const Eigen::MatrixXd>(linearisedVectors.data(), inData.dataDims, _inSize);

  size_t          n = _greedyIDs.size();
  Eigen::MatrixXd y = inputData(Eigen::all, _greedyIDs).transpose();
  Eigen::MatrixXd Cy = _cut.block(0, 0, n, n).triangularView<Eigen::Lower>() * y;
  Eigen::MatrixXd interpolationCoeffs = _cut.block(0, 0, n, n).transpose().triangularView<Eigen::Upper>() * Cy;

  for (int d = 0; d < inData.dataDims; d++) {
    outData(Eigen::seqN(d, _outSize, inData.dataDims)) = _kernelEval.transpose() * interpolationCoeffs.col(d);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData) {

  precice::profiling::Event e("map.P-greedy-cut.mapData.From" + this->input()->getName() + "To" + this->output()->getName(), profiling::Synchronize);


}

template <typename RADIAL_BASIS_FUNCTION_T>
std::string PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::getName() const {
  return "global-greedy RBF (f-cut-cpu-executor)";
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedyCutMapping<RADIAL_BASIS_FUNCTION_T>::clear() {
  _kernelEval   = Eigen::MatrixXd();
  _kernelMatrix = Eigen::MatrixXd();
  _cut          = Eigen::MatrixXd();
  _inSize       = 0;
  _outSize      = 0;
}

} // namespace mapping
} // namespace precice
