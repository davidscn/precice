#pragma once

#include <Eigen/Core>

#include <boost/container/flat_set.hpp>

#include "impl/BasisFunctions.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mapping/config/MappingConfiguration.hpp"
#include "mesh/Filter.hpp"
#include "precice/types.hpp"
#include "utils/Event.hpp"

namespace precice {

namespace mapping {

/**
 * @brief Mapping using partition of unity decomposition strategies
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class Partition {
public:
  /**
   * @brief Constructor.
   *
   * @param[in] dimensions Dimensionality of the meshes
   * @param[in] function Radial basis function used for mapping.
   * @param[in] parameter Shape parameter or support radius
   * @param[in] xDead, yDead, zDead Deactivates mapping along an axis
   */
  Partition(
      int           dimensions,
      mesh::Vertex  center,
      double        radius,
      double        parameter,
      mesh::PtrMesh inputMesh,
      mesh::PtrMesh outputMesh);

  /// Removes a computed mapping.
  void clear();

  void mapConsistent(mesh::PtrData inputData, mesh::PtrData outputData, const int);

  double computeWeight(const mesh::Vertex &v) const;

  double getInterpolatedValue(const VertexID id) const;

  bool isVertexInside(const mesh::Vertex &v) const;

  /// Indicate that the mapped data is now considered as outdated by setting _hasMappedData to false
  void                  setMappingFinished();
  unsigned int          getNumberOfInputVertices() const;
  std::array<double, 3> getCenterCoords() const;

private:
  precice::logging::Logger _log{"mapping::Partition"};

  bool _hasComputedMapping = false;
  bool _hasMappedData      = false;
  bool _emptyPartition     = false;

  mesh::Vertex _center;

  const double _radius;

  Eigen::VectorXd resultVector;

  // Stores the global IDs of the vertices so that we can apply a binary
  // search in order to query specific objects
  boost::container::flat_set<VertexID> inputIDs;
  boost::container::flat_set<VertexID> outputIDs;

  RadialBasisFctSolver _rbfSolver;
  //TODO: Can probably be removed
  RADIAL_BASIS_FUNCTION_T   _basisFunction;
  CompactThinPlateSplinesC2 _weightingFunction;

  void mapConservative(DataID inputDataID, DataID outputDataID);
};

// --------------------------------------------------- HEADER IMPLEMENTATIONS

template <typename RADIAL_BASIS_FUNCTION_T>
Partition<RADIAL_BASIS_FUNCTION_T>::Partition(
    int           dimensions,
    mesh::Vertex  center,
    double        radius,
    double        parameter,
    mesh::PtrMesh inputMesh,
    mesh::PtrMesh outputMesh)
    : _center(center), _radius(radius), _basisFunction(parameter), _weightingFunction(radius)
{
  PRECICE_DEBUG("Center coordinates: {}", _center.getCoords());
  PRECICE_DEBUG("Partition radius: {}", _radius);

  auto inIDs  = inputMesh->index().getVerticesInsideBox(center, radius);
  auto outIDs = outputMesh->index().getVerticesInsideBox(center, radius);
  if (inIDs.size() == 0 && outIDs.size() == 0) {
    _emptyPartition = true;
    return;
  }
  _emptyPartition = false;
  PRECICE_DEBUG("Source mesh size: {}", inIDs.size());
  PRECICE_DEBUG("Target mesh size: {}", outIDs.size());
  PRECICE_ASSERT(inIDs.size() > 0, "The source partition is empty whereas the target partition is non-empty.");
  // TODO: The RBF mapping requires at least four points in the source mesh
  inputIDs.insert(inIDs.begin(), inIDs.end());
  outputIDs.insert(outIDs.begin(), outIDs.end());

  _rbfSolver          = RadialBasisFctSolver{_basisFunction, *inputMesh.get(), inputIDs, *outputMesh.get(), outputIDs, {false, false}, Polynomial::ON};
  _hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
unsigned int Partition<RADIAL_BASIS_FUNCTION_T>::getNumberOfInputVertices() const
{
  return inputIDs.size();
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::array<double, 3> Partition<RADIAL_BASIS_FUNCTION_T>::getCenterCoords() const
{
  return _center.rawCoords();
}

template <typename RADIAL_BASIS_FUNCTION_T>
void Partition<RADIAL_BASIS_FUNCTION_T>::clear()
{
  PRECICE_TRACE();
  inputIDs.clear();
  outputIDs.clear();
  _hasComputedMapping = false;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void Partition<RADIAL_BASIS_FUNCTION_T>::mapConservative(DataID inputDataID, DataID outputDataID)
{
  PRECICE_CHECK(false, "Not implemented");
}

template <typename RADIAL_BASIS_FUNCTION_T>
void Partition<RADIAL_BASIS_FUNCTION_T>::mapConsistent(mesh::PtrData inputData, mesh::PtrData outputData, const int)
{
  if (_emptyPartition) {
    return;
  }
  // Serial case
  PRECICE_ASSERT(_hasComputedMapping);
  const int valueDim = inputData->getDimensions();

  const auto &localInData = inputData->values();

  Eigen::VectorXd in(_rbfSolver.getEvaluationMatrix().cols()); // rows == n
  in.setZero();

  // For every data dimension, perform mapping
  for (int dim = 0; dim < valueDim; dim++) {
    // Fill input from input data values (last polyparams entries remain zero)
    for (int i = 0; i < inputIDs.size(); i++) {
      const auto dataIndex = *(inputIDs.nth(i));
      in[i]                = localInData[dataIndex * valueDim + dim];
    }

    resultVector = _rbfSolver.solveConsistent(in, Polynomial::ON);

    // TODO: We need to transform this into something fitting for all dimensions
    PRECICE_ASSERT(valueDim == 1, "Overwriting the vector.");
  }
  _hasMappedData = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
inline double Partition<RADIAL_BASIS_FUNCTION_T>::computeWeight(const mesh::Vertex &v) const
{
  // Assume that the local interpolant and the weighting function are the same
  // TODO: We don't need to reduce the dead coordinates here as the values should reduce anyway
  std::array<double, 3> vec;
  for (unsigned int d = 0; d < vec.size(); ++d) {
    vec[d] = v.rawCoords()[d] - _center.rawCoords()[d];
  }
  double res = std::accumulate(vec.begin(), vec.end(), static_cast<double>(0.), [](auto &res, auto &val) { return res + val * val; });
  return _weightingFunction.evaluate(std::sqrt(res));
}

template <typename RADIAL_BASIS_FUNCTION_T>
inline double Partition<RADIAL_BASIS_FUNCTION_T>::getInterpolatedValue(const VertexID id) const
{
  PRECICE_ASSERT(_hasMappedData);
  PRECICE_ASSERT(outputIDs.contains(id), id);
  PRECICE_ASSERT(resultVector.size() > outputIDs.index_of(outputIDs.find(id)));
  return resultVector(outputIDs.index_of(outputIDs.find(id)));
}

template <typename RADIAL_BASIS_FUNCTION_T>
inline bool Partition<RADIAL_BASIS_FUNCTION_T>::isVertexInside(const mesh::Vertex &v) const
{
  // TODO: We don't need to reduce the dead coordinates here as the values should reduce anyway
  std::array<double, 3> vec;
  for (unsigned int d = 0; d < vec.size(); ++d) {
    vec[d] = v.rawCoords()[d] - _center.rawCoords()[d];
  }
  double res = std::accumulate(vec.begin(), vec.end(), static_cast<double>(0.), [](auto &res, auto &val) { return res + val * val; });
  return res <= std::pow(_radius, 2);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void Partition<RADIAL_BASIS_FUNCTION_T>::setMappingFinished()
{
  _hasMappedData = false;
}
// ------- Non-Member Functions ---------

// template <typename RADIAL_BASIS_FUNCTION_T>
// static Eigen::MatrixXd buildMatrixCLU(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const boost::container::flat_set<VertexID> &relevantIDs, std::vector<bool> deadAxis)
// {
//   PRECICE_ASSERT(relevantIDs.size() > 0);
//   const unsigned int inputSize  = relevantIDs.size();
//   const unsigned int dimensions = inputMesh.getDimensions();

//   const unsigned int deadDimensions = std::count(deadAxis.begin(), deadAxis.end(), true);
//   const unsigned int polyparams     = 1 + dimensions - deadDimensions;
//   const unsigned int n              = inputSize + polyparams; // Add linear polynom degrees

//   PRECICE_ASSERT(inputSize >= 1 + polyparams, inputSize);

//   Eigen::MatrixXd matrixCLU(n, n);
//   matrixCLU.setZero();

//   for (unsigned int i = 0; i < inputSize; ++i) {
//     const auto rowVertexID = *relevantIDs.nth(i);
//     for (unsigned int j = i; j < inputSize; ++j) {
//       const auto  columnVertexID = *relevantIDs.nth(j);
//       const auto &u              = inputMesh.vertices()[rowVertexID].getCoords();
//       const auto &v              = inputMesh.vertices()[columnVertexID].getCoords();
//       matrixCLU(i, j)            = basisFunction.evaluate(utils::reduceVector((u - v), deadAxis).norm());
//     }

//     const auto reduced = utils::reduceVector(inputMesh.vertices()[rowVertexID].getCoords(), deadAxis);

//     for (int dim = 0; dim < dimensions - deadDimensions; dim++) {
//       matrixCLU(i, inputSize + 1 + dim) = reduced[dim];
//     }
//     matrixCLU(i, inputSize) = 1.0;
//   }

//   matrixCLU.triangularView<Eigen::Lower>() = matrixCLU.transpose();

//   return matrixCLU;
// }

// template <typename RADIAL_BASIS_FUNCTION_T>
// static Eigen::MatrixXd buildMatrixA(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const boost::container::flat_set<VertexID> &inputIDs, const mesh::Mesh &outputMesh, const boost::container::flat_set<VertexID> &outputIDs, std::vector<bool> deadAxis)
// {
//   const unsigned int inputSize  = inputIDs.size();
//   const unsigned int outputSize = outputIDs.size();
//   const unsigned int dimensions = inputMesh.getDimensions();

//   const unsigned int deadDimensions = std::count(deadAxis.begin(), deadAxis.end(), true);
//   const unsigned int polyparams     = 1 + dimensions - deadDimensions;
//   const unsigned int n              = inputSize + polyparams; // Add linear polynom degrees

//   PRECICE_ASSERT(inputSize >= 1 + polyparams, inputSize);

//   Eigen::MatrixXd matrixA(outputSize, n);
//   matrixA.setZero();

//   // Fill _matrixA with values
//   for (unsigned int i = 0; i < outputSize; ++i) {
//     const auto rowVertexID = *outputIDs.nth(i);
//     for (unsigned int j = 0; j < inputSize; ++j) {
//       const auto  columnVertexID = *inputIDs.nth(j);
//       const auto &u              = outputMesh.vertices()[rowVertexID].getCoords();
//       const auto &v              = inputMesh.vertices()[columnVertexID].getCoords();
//       matrixA(i, j)              = basisFunction.evaluate(utils::reduceVector((u - v), deadAxis).norm());
//     }

//     const auto reduced = utils::reduceVector(outputMesh.vertices()[rowVertexID].getCoords(), deadAxis);

//     for (int dim = 0; dim < dimensions - deadDimensions; dim++) {
//       matrixA(i, inputSize + 1 + dim) = reduced[dim];
//     }
//     matrixA(i, inputSize) = 1.0;
//   }
//   return matrixA;
// }

} // namespace mapping
} // namespace precice
