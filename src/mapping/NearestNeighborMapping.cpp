#include "NearestNeighborMapping.hpp"

#include <Eigen/Core>
#include <boost/container/flat_set.hpp>
#include <functional>
#include "logging/LogMacros.hpp"
#include "mapping/RadialBasisFctSolver.hpp"
#include "mapping/impl/BasisFunctions.hpp"
#include "profiling/Event.hpp"
#include "profiling/EventUtils.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/IntraComm.hpp"
#include "utils/assertion.hpp"
namespace precice::mapping {

NearestNeighborMapping::NearestNeighborMapping(
    Constraint constraint,
    int        dimensions)
    : NearestNeighborBaseMapping(constraint, dimensions, false, "NearestNeighborMapping", "nn")
{
  if (isScaledConsistent()) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::VERTEX);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }
}

void NearestNeighborMapping::mapConservative(const time::Sample &inData, Eigen::VectorXd &outData)
{
  PRECICE_TRACE();
  precice::profiling::Event e("map." + mappingNameShort + ".mapData.From" + input()->getName() + "To" + output()->getName(), profiling::Synchronize);
  PRECICE_DEBUG("Map conservative using {}", getName());

  const Eigen::VectorXd &inputValues  = inData.values;
  Eigen::VectorXd &      outputValues = outData;

  // Data dimensions (for scalar = 1, for vectors > 1)
  const size_t    inSize = input()->nVertices();
  Eigen::VectorXd weights(output()->nVertices());
  weights.setZero();
  const int valueDimensions = inData.dataDims;

  CompactPolynomialC0 basis_function(0.5);
  for (size_t i = 0; i < inSize; i++) {
    int const outputIndex = _vertexIndices[i] * valueDimensions;

    for (int dim = 0; dim < valueDimensions; dim++) {

      const int mapOutputIndex = outputIndex + dim;
      const int mapInputIndex  = (i * valueDimensions) + dim;

      const auto &v                 = input()->vertex(mapInputIndex).rawCoords();
      auto        u                 = output()->vertex(mapOutputIndex).rawCoords();
      double      squaredDifference = computeSquaredDifference(u, v);
      auto        local_weight      = basis_function.evaluate(std::sqr(squaredDifference));
      weights[mapOutputIndex] += local_weight;

      outputValues(mapOutputIndex) += local_weight * inputValues(mapInputIndex);
    }
  }
  for (size_t i = 0; i < output()->nVertices(); i++) {
    if (weights[i] > 0)
      outputValues[i] = outputValues[i] / weights[i];
  }

  PRECICE_DEBUG("Mapped values = {}", utils::previewRange(3, outputValues));
}

void NearestNeighborMapping::mapConsistent(const time::Sample &inData, Eigen::VectorXd &outData)
{
  PRECICE_TRACE();
  precice::profiling::Event e("map." + mappingNameShort + ".mapData.From" + input()->getName() + "To" + output()->getName(), profiling::Synchronize);
  PRECICE_DEBUG("Map {} using {}", (hasConstraint(CONSISTENT) ? "consistent" : "scaled-consistent"), getName());

  const Eigen::VectorXd &inputValues  = inData.values;
  Eigen::VectorXd &      outputValues = outData;

  // Data dimensions (for scalar = 1, for vectors > 1)
  const size_t outSize         = output()->nVertices();
  const int    valueDimensions = inData.dataDims;

  for (size_t i = 0; i < outSize; i++) {
    int inputIndex = _vertexIndices[i] * valueDimensions;

    for (int dim = 0; dim < valueDimensions; dim++) {

      const int mapOutputIndex = (i * valueDimensions) + dim;
      const int mapInputIndex  = inputIndex + dim;

      outputValues(mapOutputIndex) = inputValues(mapInputIndex);
    }
  }
  PRECICE_DEBUG("Mapped values = {}", utils::previewRange(3, outputValues));
}

std::string NearestNeighborMapping::getName() const
{
  return "nearest-neighbor";
}

} // namespace precice::mapping
