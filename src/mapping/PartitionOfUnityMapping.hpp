#pragma once

#include <Eigen/Core>
#include <numeric>

#include "com/Communication.hpp"
#include "io/ExportVTU.hpp"
#include "mapping/impl/CreateClustering.hpp"
#include "mapping/impl/SphericalVertexCluster.hpp"
#include "mesh/Filter.hpp"
#include "precice/types.hpp"
#include "query/Index.hpp"
#include "utils/Event.hpp"
#include "utils/IntraComm.hpp"

namespace precice {
extern bool syncMode;

namespace mapping {

/**
 * Mapping using partition of unity decomposition strategies: The class here inherits from the Mapping
 * class and orchestrates the partitions (called vertex clusters) in order to represent a partition of unity.
 * This means in particular that the class computes the weights for the evaluation vertices and the necessary
 * association between evaluation vertices and the clusters during initialization and traverses through all
 * vertex clusters when evaluating the mapping.
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class PartitionOfUnityMapping : public Mapping {
public:
  /**
   * Constructor, which mostly sets the mesh connectivity requirements and initializes member variables.
   *
   * @param[in] constraint Specifies mapping to be consistent or conservative.
   * @param[in] dimension Dimensionality of the meshes
   * @param[in] function Radial basis function type used in interpolation
   * @param[in] polynomial The handling of the polynomial in the RBF system. Valid choices are 'off' and 'separate'
   * @param[in] verticesPerCluster Target number of vertices to be clustered together
   * @param[in] relativeOverlap Overlap between clusters: The parameter here determines the distance between two cluster
   * centers, given the cluster radius (already determined through \p verticesPerCluster ). A value of 1 would correspond
   * to no distance between cluster centers (i.e. completely overlapping clusters), 0 to distance of 2 x radius between
   * clusters centers.
   * @param[in] projectToInput if enabled, places the cluster centers at the closest vertex of the input mesh.
   * See also \ref mapping::impl::createClustering()
   */
  PartitionOfUnityMapping(
      Mapping::Constraint     constraint,
      int                     dimension,
      RADIAL_BASIS_FUNCTION_T function,
      std::array<bool, 3>     deadAxis,
      Polynomial              polynomial,
      unsigned int            verticesPerCluster,
      double                  relativeOverlap,
      bool                    projectToInput);

  /**
   * Computes the clustering for the partition of unity method and fills the \p _clusters vector,
   * which allows to travers through all vertex cluster computed. Each vertex cluster in the vector
   * directly computes local mapping matrices and matrix decompositions.
   * In addition, the method computes the normalized weights (Shepard's method) for the partition
   * of unity method and stores them directly in each relevant vertex cluster.
   * In debug mode, the function also exports the partition centers as a separate mesh for visualization
   * purpose.
   */
  virtual void computeMapping() override;

  /// Clears a computed mapping by deleting the content of the \p _clusters vector.
  virtual void clear() override;

  /// tag the vertices required for the mapping
  virtual void tagMeshFirstRound() override;

  /// nothing to do here
  virtual void tagMeshSecondRound() override;

private:
  /// logger, as usual
  precice::logging::Logger _log{"mapping::PartitionOfUnityMapping"};

  /// main data container storing all the clusters, which need to be solved individually
  std::vector<SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T>> _clusters;

  /// Radial basis function type used in interpolation
  RADIAL_BASIS_FUNCTION_T _basisFunction;

  /// Input parameters provided by the user for the clustering algorithm:

  /// target number of input vertices for each cluster
  const unsigned int _verticesPerCluster;

  /// overlap of vertex clusters
  const double _relativeOverlap;

  /// toggles whether we project the cluster centers to the input mesh
  const bool _projectToInput;

  /// derived parameter based on the input above: the radius of each cluster
  double averageClusterRadius = 0;

  /// true if the mapping along some axis should be ignored
  /// has currently only dim x false entries, as integrated polynomials are irrelevant
  std::vector<bool> _deadAxis;

  /// polynomial treatment of the RBF system
  Polynomial _polynomial;

  /// @copydoc Mapping::mapConservative
  virtual void mapConservative(DataID inputDataID, DataID outputDataID) override;

  /// @copydoc Mapping::mapConsistent
  virtual void mapConsistent(DataID inputDataID, DataID outputDataID) override;

  /// export the center vertices of all clusters as a mesh with some additional data on it such as vertex count
  /// only enabled in debug builds and mainly for debugging purpose
  void exportClusterCentersAsVTU(mesh::Mesh &centers);
};

template <typename RADIAL_BASIS_FUNCTION_T>
PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::PartitionOfUnityMapping(
    Mapping::Constraint     constraint,
    int                     dimension,
    RADIAL_BASIS_FUNCTION_T function,
    std::array<bool, 3>     deadAxis,
    Polynomial              polynomial,
    unsigned int            verticesPerCluster,
    double                  relativeOverlap,
    bool                    projectToInput)
    : Mapping(constraint, dimension),
      _basisFunction(function), _verticesPerCluster(verticesPerCluster), _relativeOverlap(relativeOverlap), _projectToInput(projectToInput), _polynomial(polynomial)
{
  PRECICE_ASSERT(this->getDimensions() <= 3);
  PRECICE_CHECK(_polynomial != Polynomial::ON, "Integrated polynomial is not supported for partition of unity data mappings.");
  PRECICE_ASSERT(_relativeOverlap < 1, "The relative overlap has to be smaller than one.");
  PRECICE_ASSERT(_verticesPerCluster > 0, "The number of vertices per cluster has to be greater zero.");

  if (isScaledConsistent()) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::VERTEX);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }

  _deadAxis.clear();
  std::copy_n(deadAxis.begin(), getDimensions(), std::back_inserter(_deadAxis));
  if (getDimensions() == 2 && deadAxis[2]) {
    PRECICE_WARN("Setting the z-axis to dead on a 2-dimensional problem has no effect. Please remove the respective mapping's \"z-dead\" attribute.");
  }
  PRECICE_CHECK(std::any_of(_deadAxis.begin(), _deadAxis.end(), [](const auto &ax) { return ax == false; }), "You cannot set all axes to dead for an RBF mapping. Please remove one of the respective mapping's \"x-dead\", \"y-dead\", or \"z-dead\" attributes.");
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping()
{
  PRECICE_TRACE();

  precice::utils::Event e("map.pou.computeMapping.From" + this->input()->getName() + "To" + this->output()->getName(), precice::syncMode);

  // Recompute the whole clustering
  PRECICE_ASSERT(!this->_hasComputedMapping, "Please clear the mapping before recomputing.");

  mesh::PtrMesh inMesh;
  mesh::PtrMesh outMesh;
  if (this->hasConstraint(Mapping::CONSERVATIVE)) {
    inMesh  = this->output();
    outMesh = this->input();
  } else { // Consistent or scaled consistent
    inMesh  = this->input();
    outMesh = this->output();
  }

  // Step 1: get a tentative clustering consisting of centers and a radius from one of the available algorithms
  auto [averageClusterRadius_, centerCandidates] = impl::createClustering(inMesh, outMesh, _relativeOverlap, _verticesPerCluster, _projectToInput);

  averageClusterRadius = averageClusterRadius_;
  PRECICE_ASSERT(averageClusterRadius > 0 || inMesh->vertices().size() == 0 || outMesh->vertices().size() == 0);

  // Step 2: check, which of the resulting clusters are non-empty and register the cluster centers in a mesh
  // Here, the VertexCluster computes the matrix decompositions directly in case the cluster is non-empty
  mesh::Mesh centerMesh("pou-centers-" + inMesh->getName(), this->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
  auto &     meshVertices = centerMesh.vertices();

  for (const auto &c : centerCandidates) {
    // We cannot simply copy the vertex from the container in order to fill the vertices of the centerMesh, as the vertexID of each center needs to match the index
    // of the cluster within the _clusters vector. That's required for the indexing further down and asserted below
    const VertexID                                  vertexID = meshVertices.size();
    mesh::Vertex                                    center(c.getCoords(), vertexID);
    SphericalVertexCluster<RADIAL_BASIS_FUNCTION_T> cluster(center, averageClusterRadius, _basisFunction, _deadAxis, _polynomial, inMesh, outMesh);

    // Consider only non-empty clusters
    if (!cluster.empty()) {
      PRECICE_ASSERT(center.getID() == _clusters.size(), center.getID(), _clusters.size());
      meshVertices.emplace_back(std::move(center));
      _clusters.emplace_back(std::move(cluster));
    }
  }
  // Log the average number of resulting clusters
  PRECICE_INFO("Number of total clusters (final): {}", _clusters.size());

  if (_clusters.size() > 0) {
    unsigned int nVertices   = std::accumulate(_clusters.begin(), _clusters.end(), static_cast<unsigned int>(0), [](auto &acc, auto &val) { return acc += val.getNumberOfInputVertices(); });
    unsigned int maxVertices = std::max_element(_clusters.begin(), _clusters.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices();
    unsigned int minVertices = std::min_element(_clusters.begin(), _clusters.end(), [](auto &v1, auto &v2) { return v1.getNumberOfInputVertices() < v2.getNumberOfInputVertices(); })->getNumberOfInputVertices();
    PRECICE_INFO("Average number of vertices per cluster {}", nVertices / _clusters.size());
    PRECICE_INFO("Maximum number of vertices per cluster {}", maxVertices);
    PRECICE_INFO("Minimum number of vertices per cluster {}", minVertices);
  }

  // Log a bounding box of the center mesh
  centerMesh.computeBoundingBox();
  PRECICE_INFO("Bounding Box of the cluster centers {}", centerMesh.getBoundingBox());

  // Step 3: index the clusters / the center mesh in order to define the output vertex -> cluster ownership
  // the ownership is required to compute the normalized partition of unity weights (Step 4)
  query::Index clusterIndex(centerMesh);
  // Find all clusters the output vertex lies in, i.e., find all cluster centers which have the distance of a cluster radius from the given output vertex
  // Here, we do this using the RTree on the clusterMesh: VertexID (queried from the centersMesh) == clusterID, by construction above.
  // Note: this could also be done on-the-fly in the map data phase for dynamic queries, which would require to make the mesh as well as the indexTree member variables.
  PRECICE_DEBUG("Computing cluster-vertex association");
  for (const auto &vertex : outMesh->vertices()) {
    auto clusterIDs = clusterIndex.getVerticesInsideBox(vertex, averageClusterRadius);
    // Consider the case where we didn't find any cluster (meshes don't match very well)
    if (clusterIDs.size() == 0) {
      PRECICE_WARN("Output vertex {} could not be assigned to a cluster. This means that the meshes probably do not match well geometry-wise.", vertex.getCoords());
      // TODO: Think about a proper way to handle this case, maybe set all radii to distance(v, closestvertex)?
      clusterIDs.emplace_back(clusterIndex.getClosestVertex(vertex.getCoords()).index);
    }

    // Step 4: compute the normalized weights of each output vertex for each partition
    PRECICE_ASSERT(clusterIDs.size() > 0, "No cluster found for vertex {}", vertex.getCoords());

    // Step 4a: compute the weight in each partition individually and store them in 'weights'
    std::vector<double> weights(clusterIDs.size());
    std::transform(clusterIDs.cbegin(), clusterIDs.cend(), weights.begin(), [&](const auto &ids) { return _clusters[ids].computeWeight(vertex); });
    double weightSum = std::accumulate(weights.begin(), weights.end(), static_cast<double>(0.));
    // TODO: This covers the edge case of vertices being at the edge of (several) clusters
    // In case the sum is equal to zero, we assign equal weights for all clusters
    if (!(weightSum > 0)) {
      PRECICE_ASSERT(weights.size() > 0);
      std::for_each(weights.begin(), weights.end(), [&weights](auto &w) { w = 1 / weights.size(); });
      weightSum = 1;
    }
    PRECICE_DEBUG("Weight sum {}", weightSum);
    PRECICE_DEBUG("Clusters {}", clusterIDs);
    PRECICE_DEBUG("V coords {}", vertex.getCoords());
    PRECICE_ASSERT(weightSum > 0);

    // Step 4b: scale the weight using the weight sum and store the normalized weight in all associated clusters
    for (unsigned int i = 0; i < clusterIDs.size(); ++i) {
      PRECICE_ASSERT(clusterIDs[i] < _clusters.size());
      _clusters[clusterIDs[i]].setNormalizedWeight(weights[i] / weightSum, vertex.getID());
    }
  }

// Add a VTK export for visualization purposes
#ifndef NDEBUG
  exportClusterCentersAsVTU(centerMesh);
#endif

  this->_hasComputedMapping = true;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(DataID inputDataID, DataID outputDataID)
{
  PRECICE_TRACE(inputDataID, outputDataID);

  precice::utils::Event e("map.pou.mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  // Execute the actual mapping evaluation in all clusters
  // 1. Reset all output data values as we accumulate data in all clusters independent
  output()->data(outputDataID)->values().setZero();

  // 2. Iterate over all clusters and accumulate the result in the output data
  std::for_each(_clusters.begin(), _clusters.end(), [&](auto &p) { p.mapConservative(input()->data(inputDataID),
                                                                                     output()->data(outputDataID)); });
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(DataID inputDataID, DataID outputDataID)
{
  PRECICE_TRACE(inputDataID, outputDataID);

  precice::utils::Event e("map.pou.mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  // Execute the actual mapping evaluation in all clusters
  // 1. Reset all output data values as we accumulate data in all clusters independent
  output()->data(outputDataID)->values().setZero();

  // 2. Execute the actual mapping evaluation in all vertex clusters and accumulate the data
  std::for_each(_clusters.begin(), _clusters.end(), [&](auto &p) { p.mapConsistent(input()->data(inputDataID),
                                                                                   output()->data(outputDataID)); });
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

  // TODO: Check again the tagging in combination with the partition construction (which mesh to use)
  // In order to construct the local partitions, we need all vertices with a distance of 2 x radius,
  // as the relevant partitions centers have a maximum distance of radius, and the proper construction of the
  // interpolant requires all vertices with a distance of radius from the center.
  auto bb = outMesh->getBoundingBox();

  if (averageClusterRadius == 0)
    averageClusterRadius = impl::estimateClusterRadius(_verticesPerCluster, filterMesh, bb);

  // @TODO: This assert is not completely right, as it checks all dimensions for non-emptyness (which might not be the case).
  // However, with the current BB implementation, the expandBy function will just do nothing.
  PRECICE_ASSERT(!bb.empty());
  PRECICE_ASSERT(averageClusterRadius > 0);
  // Now we extend the bounding box by the radius
  bb.expandBy(1 * averageClusterRadius);

  // ... and tag all affected vertices
  auto verticesNew = filterMesh->index().getVerticesInsideBox(bb);

  std::for_each(verticesNew.begin(), verticesNew.end(), [&filterMesh](VertexID v) { filterMesh->vertices()[v].tag(); });
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshSecondRound()
{
  // Probably nothing to be done here. There is no global ownership for matrix entries required and we tag all potentially locally relevant vertices already in the first round.
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::exportClusterCentersAsVTU(mesh::Mesh &centerMesh)
{
  PRECICE_TRACE();

  auto dataRadius      = centerMesh.createData("radius", 1, -1);
  auto dataCardinality = centerMesh.createData("number-of-vertices", 1, -1);
  centerMesh.allocateDataValues();
  dataRadius->values().fill(averageClusterRadius);
  for (unsigned int i = 0; i < _clusters.size(); ++i) {
    dataCardinality->values()[i] = static_cast<double>(_clusters[i].getNumberOfInputVertices());
  }

  // We have to create the global offsets in order to export things in parallel
  if (utils::IntraComm::isSecondary()) {
    // send number of vertices
    PRECICE_DEBUG("Send number of vertices: {}", centerMesh.vertices().size());
    int numberOfVertices = centerMesh.vertices().size();
    utils::IntraComm::getCommunication()->send(numberOfVertices, 0);

    // receive vertex offsets
    mesh::Mesh::VertexOffsets vertexOffsets;
    utils::IntraComm::getCommunication()->broadcast(vertexOffsets, 0);
    PRECICE_DEBUG("My vertex offsets: {}", vertexOffsets);
    PRECICE_ASSERT(centerMesh.getVertexOffsets().empty());
    centerMesh.setVertexOffsets(std::move(vertexOffsets));
  } else if (utils::IntraComm::isPrimary()) {

    mesh::Mesh::VertexOffsets vertexOffsets(utils::IntraComm::getSize());
    vertexOffsets[0] = centerMesh.vertices().size();

    // receive number of secondary vertices and fill vertex offsets
    for (int secondaryRank : utils::IntraComm::allSecondaryRanks()) {
      int numberOfSecondaryRankVertices = -1;
      utils::IntraComm::getCommunication()->receive(numberOfSecondaryRankVertices, secondaryRank);
      PRECICE_ASSERT(numberOfSecondaryRankVertices >= 0);
      vertexOffsets[secondaryRank] = numberOfSecondaryRankVertices + vertexOffsets[secondaryRank - 1];
    }

    // broadcast vertex offsets
    PRECICE_DEBUG("My vertex offsets: {}", centerMesh.getVertexOffsets());
    utils::IntraComm::getCommunication()->broadcast(vertexOffsets);
    centerMesh.setVertexOffsets(std::move(vertexOffsets));
  }

  io::ExportVTU exporter;
  exporter.doExport(centerMesh.getName(), "exports", centerMesh);
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PartitionOfUnityMapping<RADIAL_BASIS_FUNCTION_T>::clear()
{
  PRECICE_TRACE();
  _clusters.clear();
  // TODO: Don't reset this here
  averageClusterRadius      = 0;
  this->_hasComputedMapping = false;
}
} // namespace mapping
} // namespace precice
