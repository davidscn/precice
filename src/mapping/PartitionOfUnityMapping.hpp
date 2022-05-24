#pragma once

#include <Eigen/Core>
#include <numeric>

#include "com/CommunicateMesh.hpp"
#include "com/Communication.hpp"
#include "impl/BasisFunctions.hpp"
#include "mapping/RadialBasisFctBaseMapping.hpp"
#include "mesh/Filter.hpp"
#include "precice/types.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/Event.hpp"
#include "utils/IntraComm.hpp"

#include "mapping/Partition.hpp"

namespace precice {
extern bool syncMode;

namespace mapping {

/**
 * @brief Mapping using partition of unity decomposition strategies
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class PartitionOfUnityMapping : public Mapping {
public:
  /**
   * @brief Constructor.
   *
   * @param[in] constraint Specifies mapping to be consistent or conservative.
   * @param[in] dimensions Dimensionality of the meshes
   * @param[in] parameter shape parameter or support radius of the interpolation RBF
   * @param[in] function Radial basis function used for mapping.
   * @param[in] xDead, yDead, zDead Deactivates mapping along an axis
   */
  PartitionOfUnityMapping(
      Mapping::Constraint constraint,
      int                 dimensions,
      double              parameter,
      unsigned int        verticesPerPartition,
      double              relativeOverlap,
      std::array<bool, 3> deadAxis);

  /// Computes the mapping coefficients from the in- and output mesh.
  virtual void computeMapping() override;

  /// Removes a computed mapping.
  virtual void clear() override;

  virtual void tagMeshFirstRound() override;

  virtual void tagMeshSecondRound() override;

private:
  precice::logging::Logger _log{"mapping::PartitionOfUnityMapping"};

  std::vector<Partition<RADIAL_BASIS_FUNCTION_T>> _partitions;

  // Shape parameter or support radius for the RBF interpolant,
  // only required for the Partition instantiation
  const double _parameter;

  // Input parameter
  const unsigned int _verticesPerPartition;
  const double       _relativeOverlap;

  // Derived parameter
  double                      averagePartitionRadius = 0;
  std::array<unsigned int, 3> nPartitions{1, 1, 1};

  int deadDimensions{};

  /// @copydoc Mapping::mapConservative
  virtual void mapConservative(DataID inputDataID, DataID outputDataID) override;

  /// @copydoc Mapping::mapConsistent
  virtual void mapConsistent(DataID inputDataID, DataID outputDataID) override;

  void
  estimatePartitioning();
};

// --------------------------------------------------- HEADER IMPLEMENTATIONS

template <typename RADIAL_BASIS_FUNCTION_T>
PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::PartitionOfUnityMapping(
    Mapping::Constraint constraint,
    int                 dimensions,
    double              parameter,
    unsigned int        verticesPerPartition,
    double              relativeOverlap,
    std::array<bool, 3> deadAxis)
    : Mapping(constraint, dimensions),
      _parameter(parameter), _verticesPerPartition(verticesPerPartition), _relativeOverlap(relativeOverlap)
{
  PRECICE_CHECK(_relativeOverlap < 1, "The relative overlap has to be smaller than one.");
  PRECICE_CHECK(_verticesPerPartition > 0, "The number of vertices per partition has to be greater zero.");

  if (constraint == SCALEDCONSISTENT) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::VERTEX);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }
  deadDimensions = std::count(deadAxis.begin(), deadAxis.end(), true);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::estimatePartitioning()
{
  PRECICE_TRACE();
  mesh::PtrMesh filterMesh, outMesh;
  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    filterMesh = this->output(); // remote
    outMesh    = this->input();  // local
  } else {
    filterMesh = this->input();  // remote
    outMesh    = this->output(); // local
  }
  // Get the number of points in the input mesh lying within the output mesh region
  // in order to estimate the global number of vertices this rank has to compute the
  // mapping on
  PRECICE_DEBUG("Relative overlap: {}", _relativeOverlap);
  PRECICE_DEBUG("Vertices per partition: {}", _verticesPerPartition);
  // 1. Get the global bounding box of the output mesh
  outMesh->computeBoundingBox();
  auto               bb         = outMesh->getBoundingBox();
  auto               vertices   = filterMesh->index().getVerticesInsideBox(bb);
  const unsigned int vertexSize = vertices.size();

  // 2. Based on the input parameter _verticesPerPartition and overlap, estaimte the number
  // of partitions and the radius
  PRECICE_ASSERT(_relativeOverlap < 1);
  PRECICE_ASSERT(_verticesPerPartition > 0);
  PRECICE_INFO("Input mesh size: {}", vertexSize);
  unsigned int nTotalPartitions = std::max(1., (vertexSize / _verticesPerPartition) * (1. / (1 - _relativeOverlap)));
  PRECICE_INFO("Number of total partitions: {}", nTotalPartitions);

  // Get the edge length of the bounding box in each direction
  std::vector<double> edgeLength;
  for (unsigned int d = 0; d < bb.getDimension(); ++d) {
    edgeLength.emplace_back(bb.getEdgeLength(d));
  }
  PRECICE_ASSERT(bb.getDimension() == 2, "Not implemented.");

  // Assume uniform distribution in the bounding box:
  // xEdge/yEdge = nPartitionsX/nPartitionsY, nPartitionsX * nPartitionsY = nPartitions
  // --> nPartitionsX = sqrt(nPartitions* (xEdge/yEdge))

  // Values are ceiled
  nPartitions[0] = std::ceil(std::max(1., std::sqrt(nTotalPartitions * (edgeLength[0] / edgeLength[1]))));
  nPartitions[1] = std::ceil(std::max(1., std::sqrt(nTotalPartitions * (edgeLength[1] / edgeLength[0]))));

  PRECICE_INFO("Partition distribution: {}", nPartitions);

  // Compute the radius based on the edge length and the number of partitions
  // 0.5 since we use a radius here instead of a diameter
  PRECICE_ASSERT(!bb.empty());
  const double distance  = std::max(edgeLength[0] / nPartitions[0], edgeLength[1] / nPartitions[1]);
  averagePartitionRadius = std::pow(_relativeOverlap, 2) * distance + _relativeOverlap * 0.5 * distance + 0.5 * distance;
  PRECICE_INFO("Partition Radius: {}", averagePartitionRadius);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshFirstRound()
{
  PRECICE_TRACE();
  mesh::PtrMesh filterMesh, outMesh;
  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    filterMesh = this->output(); // remote
    outMesh    = this->input();  // local
  } else {
    filterMesh = this->input();  // remote
    outMesh    = this->output(); // local
  }

  if (outMesh->vertices().empty())
    return; // Ranks not at the interface should never hold interface vertices

  if (averagePartitionRadius == 0) {
    estimatePartitioning();
  }
  PRECICE_DEBUG("Partition Radius: {}", averagePartitionRadius);

  auto bb = outMesh->getBoundingBox();
  // Now we extend the bounding box by the radius
  bb.expandBy(averagePartitionRadius);

  // ... and tag all affected vertices
  auto verticesNew = filterMesh->index().getVerticesInsideBox(bb);

  std::for_each(verticesNew.begin(), verticesNew.end(), [&filterMesh](size_t v) { filterMesh->vertices()[v].tag(); });
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshSecondRound()
{
  // Nothing required here
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping()
{
  PRECICE_TRACE();

  precice::utils::Event e("map.pou.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), precice::syncMode);

  // Recompute the whole partitioning
  _partitions.clear();

  PRECICE_ASSERT(this->input()->getDimensions() == this->output()->getDimensions(),
                 this->input()->getDimensions(), this->output()->getDimensions());
  PRECICE_ASSERT(this->getDimensions() == this->output()->getDimensions(),
                 this->getDimensions(), this->output()->getDimensions());

  mesh::PtrMesh inMesh;
  mesh::PtrMesh outMesh;

  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    inMesh  = this->output();
    outMesh = this->input();
  } else { // Consistent or scaled consistent
    inMesh  = this->input();
    outMesh = this->output();
  }

  if (averagePartitionRadius == 0) {
    estimatePartitioning();
  }

  // Compute the individual partitions
  {
    using mesh::Vertex;
    // 1. Determine the centers of the partitions
    const auto &        bb  = outMesh->getBoundingBox();
    const int           dim = 2;
    std::vector<double> centerCoords(dim);
    const double        center_x = bb.getEdgeLength(0) / nPartitions[0];
    const double        center_y = bb.getEdgeLength(1) / nPartitions[1];

    // TODO: Partitions with no output vertex should not be constructed
    // start with the (bottom left) corner
    centerCoords[0] = bb.getDirectionsCoordinates(0).first + 0.5 * center_x;
    for (unsigned int x = 0; x < nPartitions[0]; ++x) {
      centerCoords[1] = bb.getDirectionsCoordinates(1).first + 0.5 * center_y;
      for (unsigned int y = 0; y < nPartitions[1]; ++y) {
        Vertex center(centerCoords, -1);
        _partitions.emplace_back(Partition<RADIAL_BASIS_FUNCTION_T>(inMesh->getDimensions(), center, averagePartitionRadius, _parameter, inMesh, outMesh));
        // for all x coordinates, iterate over the corresponding y coordinates
        // 2 since we are dealing with the radius from both partitions
        centerCoords[1] += center_y;
      }
      centerCoords[0] += center_x;
    }
  }
  unsigned int nVertices = std::accumulate(_partitions.begin(), _partitions.end(), static_cast<unsigned int>(0), [](auto &acc, auto &val) { return acc += val.getNumberOfInputVertices(); });
  double       min_x     = std::min_element(_partitions.begin(), _partitions.end(), [](auto &p1, auto &p2) { return p1.getCenterCoords()[0] < p2.getCenterCoords()[0]; })->getCenterCoords()[0];
  double       min_y     = std::min_element(_partitions.begin(), _partitions.end(), [](auto &p1, auto &p2) { return p1.getCenterCoords()[1] < p2.getCenterCoords()[1]; })->getCenterCoords()[1];
  double       max_x     = std::max_element(_partitions.begin(), _partitions.end(), [](auto &p1, auto &p2) { return p1.getCenterCoords()[0] < p2.getCenterCoords()[0]; })->getCenterCoords()[0];
  double       max_y     = std::max_element(_partitions.begin(), _partitions.end(), [](auto &p1, auto &p2) { return p1.getCenterCoords()[1] < p2.getCenterCoords()[1]; })->getCenterCoords()[1];

  PRECICE_INFO("Average number of vertices {}", nVertices / _partitions.size());
  PRECICE_INFO("Bounding Box x {} to {}, y {} to {}", min_x, max_x, min_y, max_y);
  this->_hasComputedMapping = true;
  PRECICE_DEBUG("Compute Mapping is Completed.");
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::clear()
{
  PRECICE_TRACE();
  _partitions.clear();
  this->_hasComputedMapping = false;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(DataID inputDataID, DataID outputDataID)
{
  PRECICE_CHECK(false, "Not implemented");
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(DataID inputDataID, DataID outputDataID)
{
  precice::utils::Event e("map.pou.mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  // More detailed measurements
  precice::utils::Event e_evaluate("map.pou.mapConsistent.Evaluate.From" + this->input()->getName() + "To" + this->output()->getName(), precice::syncMode);
  PRECICE_TRACE(inputDataID, outputDataID);
  // Execute the actual mapping evaluation in all partitions
  // @TODO: Use lazy mapping execution on demand later on
  std::for_each(_partitions.begin(), _partitions.end(), [&](auto &p) { p.mapConsistent(input()->data(inputDataID),
                                                                                       output()->data(outputDataID),
                                                                                       1 + getDimensions() - deadDimensions); });
  e_evaluate.stop();
  precice::utils::Event e_exec("map.pou.mapConsistent.Execute.From" + this->input()->getName() + "To" + this->output()->getName(), precice::syncMode);

  // Iterate over all vertices and update the output data
  for (const auto &v : this->output()->vertices()) {
    // 1. Find all partitions the output vertex lies in
    std::vector<unsigned int> partitionIDs;
    for (unsigned int p = 0; p < _partitions.size(); ++p) {
      if (_partitions[p].isVertexInside(v)) {
        partitionIDs.emplace_back(p);
      }
    }
    PRECICE_ASSERT(partitionIDs.size() > 0, "No partition found for vertex v ", v.getID());

    // 2. In each partition, gather the weights
    std::vector<double> weights(partitionIDs.size());
    std::transform(partitionIDs.cbegin(), partitionIDs.cend(), weights.begin(), [&](const auto &ids) { return _partitions[ids].computeWeight(v); });
    double weightSum = std::accumulate(weights.begin(), weights.end(), static_cast<double>(0.));
    // TODO: This covers the edge case of vertices being at the edge of (several) partitions
    // In case the sum is equal to zero, we assign equal weights for all partitions
    if (!weightSum > 0) {
      PRECICE_ASSERT(weights.size() > 0);
      std::for_each(weights.begin(), weights.end(), [&weights](auto &w) { w = 1 / weights.size(); });
      weightSum = 1;
    }
    PRECICE_DEBUG("Weight sum {}", weightSum);
    PRECICE_DEBUG("Partitions {}", partitionIDs);
    PRECICE_DEBUG("V coords {}", v.getCoords());
    PRECICE_ASSERT(weightSum > 0);
    // TODO: Transform into a vector for vector data
    double result = 0;
    for (unsigned int i = 0; i < partitionIDs.size(); ++i) {
      result += (_partitions[partitionIDs[i]].getInterpolatedValue(v.getID()) * weights[i]) / weightSum;
    }
    PRECICE_DEBUG("Result {}", result);

    // 3. Update the output data
    const int dim                                                 = 1;
    this->output()->data(outputDataID)->values()[v.getID() * dim] = result;
    // std::copy_n(std::begin(result), dim, &this->output()->data(outputDataID)->values()[vertexID * dim])
  }

  // Set mapping finished
  std::for_each(_partitions.begin(), _partitions.end(), [&](auto &p) { p.setMappingFinished(); });
}
} // namespace mapping
} // namespace precice
