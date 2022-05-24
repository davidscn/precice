#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include "logging/Logger.hpp"
#include "mapping/Mapping.hpp"
#include "mapping/PartitionOfUnityMapping.hpp"
#include "mapping/impl/BasisFunctions.hpp"
#include "mesh/Data.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Utils.hpp"
#include "mesh/Vertex.hpp"
#include "testing/TestContext.hpp"
#include "testing/Testing.hpp"

using namespace precice;
using namespace precice::mesh;
using namespace precice::mapping;
using namespace precice::testing;
using precice::testing::TestContext;

BOOST_AUTO_TEST_SUITE(MappingTests)
BOOST_AUTO_TEST_SUITE(PartitionOfUnityMapping)

void addGlobalIndex(mesh::PtrMesh &mesh, int offset = 0)
{
  for (mesh::Vertex &v : mesh->vertices()) {
    v.setGlobalIndex(v.getID() + offset);
  }
}

void testSerialScaledConsistent(mesh::PtrMesh inMesh, mesh::PtrMesh outMesh, mesh::PtrData inData, mesh::PtrData outData)
{
  auto inputIntegral  = mesh::integrate(inMesh, inData);
  auto outputIntegral = mesh::integrate(outMesh, outData);

  for (int dim = 0; dim < inputIntegral.size(); ++dim) {
    BOOST_TEST(inputIntegral(dim) == outputIntegral(dim));
  }
}

BOOST_AUTO_TEST_SUITE(Serial)

void perform2DTestConsistentMapping(Mapping &mapping)
{
  int dimensions = 2;
  using Eigen::Vector2d;

  // Create mesh to map from
  mesh::PtrMesh inMesh(new mesh::Mesh("InMesh", dimensions, testing::nextMeshID()));
  mesh::PtrData inData   = inMesh->createData("InData", 1, 0_dataID);
  int           inDataID = inData->getID();
  inMesh->createVertex(Vector2d(0.0, 0.0));
  inMesh->createVertex(Vector2d(1.0, 0.0));
  inMesh->createVertex(Vector2d(1.0, 1.0));
  inMesh->createVertex(Vector2d(0.0, 1.0));

  inMesh->createVertex(Vector2d(2.0, 0.0));
  inMesh->createVertex(Vector2d(3.0, 0.0));
  inMesh->createVertex(Vector2d(3.0, 1.0));
  inMesh->createVertex(Vector2d(2.0, 1.0));

  inMesh->createVertex(Vector2d(4.0, 0.0));
  inMesh->createVertex(Vector2d(5.0, 0.0));
  inMesh->createVertex(Vector2d(5.0, 1.0));
  inMesh->createVertex(Vector2d(4.0, 1.0));

  inMesh->allocateDataValues();
  addGlobalIndex(inMesh);

  auto &values = inData->values();
  values << 1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 6.0, 6.0, 5.0;

  // Create mesh to map to
  mesh::PtrMesh outMesh(new mesh::Mesh("OutMesh", dimensions, testing::nextMeshID()));
  mesh::PtrData outData   = outMesh->createData("OutData", 1, 1_dataID);
  int           outDataID = outData->getID();
  mesh::Vertex &vertex    = outMesh->createVertex(Vector2d(0, 0));
  mesh::Vertex &vertex1   = outMesh->createVertex(Vector2d(3.5, 0.5));
  mesh::Vertex &vertex2   = outMesh->createVertex(Vector2d(2.5, 0.5));
  outMesh->createVertex(Vector2d(0, 0));
  outMesh->createVertex(Vector2d(5, 1));
  outMesh->allocateDataValues();
  addGlobalIndex(outMesh);

  // Setup mapping with mapping coordinates and geometry used
  mapping.setMeshes(inMesh, outMesh);
  BOOST_TEST(mapping.hasComputedMapping() == false);

  vertex.setCoords(Vector2d(0.0, 0.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  double value  = outData->values()(0);
  double value1 = outData->values()(1);
  double value2 = outData->values()(2);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 1.0);
  BOOST_TEST(value1 == 4.5);
  BOOST_TEST(value2 == 3.5);

  vertex.setCoords(Vector2d(0.0, 0.5));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 1.0);

  vertex.setCoords(Vector2d(0.0, 1.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 1.0);

  vertex.setCoords(Vector2d(1.0, 0.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 2.0);

  vertex.setCoords(Vector2d(1.0, 0.5));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 2.0);

  vertex.setCoords(Vector2d(1.0, 1.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 2.0);

  vertex.setCoords(Vector2d(0.5, 0.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 1.5);

  vertex.setCoords(Vector2d(0.5, 0.5));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 1.5);

  vertex.setCoords(Vector2d(0.5, 1.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  value = outData->values()(0);
  BOOST_TEST(mapping.hasComputedMapping() == true);
  BOOST_TEST(value == 1.5);
}

BOOST_AUTO_TEST_CASE(MapPoU)
{
  PRECICE_TEST(1_rank);
  std::array<bool, 3> deadAxis;
  deadAxis.fill(false);
  mapping::PartitionOfUnityMapping<CompactPolynomialC0> consistentMap2D(Mapping::CONSISTENT, 2, 3., 5, 0.4, deadAxis);
  perform2DTestConsistentMapping(consistentMap2D);
}

BOOST_AUTO_TEST_SUITE_END() // Serial

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
