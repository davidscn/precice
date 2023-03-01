#ifndef PRECICE_NO_MPI

#include "helpers.hpp"
#include "testing/Testing.hpp"

#include "mesh/Utils.hpp"
#include "precice/SolverInterface.hpp"
#include "precice/impl/SolverInterfaceImpl.hpp"

void testMappingNearestProjection(bool defineEdgesExplicitly, bool useBulkFunctions, const std::string configFile, const TestContext &context)
{
  using Eigen::Vector3d;

  const double z = 0.3;

  // MeshOne
  Vector3d coordOneA{0.0, 0.0, z};
  Vector3d coordOneB{1.0, 0.0, z};
  Vector3d coordOneC{1.0, 1.0, z};
  Vector3d coordOneD{0.0, 1.0, z};
  double   valOneA = 1.0;
  double   valOneB = 3.0;
  double   valOneC = 5.0;
  double   valOneD = 7.0;

  // MeshTwo
  Vector3d coordTwoA{0.0, 0.0, z + 0.1};               // Maps to vertex A
  Vector3d coordTwoB{0.0, 0.5, z - 0.01};              // Maps to edge AD
  Vector3d coordTwoC{2.0 / 3.0, 1.0 / 3.0, z + 0.001}; // Maps to triangle ABC
  // This corresponds to the point C from mesh two on the triangle ABC on mesh one.
  Vector3d barycenterABC{0.3798734633239789, 0.24025307335204216, 0.3798734633239789};
  double   expectedValTwoA = 1.0;
  double   expectedValTwoB = 4.0;
  double   expectedValTwoC = Vector3d{valOneA, valOneB, valOneC}.dot(barycenterABC);

  if (context.isNamed("SolverOne")) {
    precice::SolverInterface interface("SolverOne", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshID = "MeshOne";

    // Setup mesh one.
    int idA = interface.setMeshVertex(meshID, coordOneA.data());
    int idB = interface.setMeshVertex(meshID, coordOneB.data());
    int idC = interface.setMeshVertex(meshID, coordOneC.data());
    int idD = interface.setMeshVertex(meshID, coordOneD.data());

    if (defineEdgesExplicitly) {
      if (useBulkFunctions) {
        std::vector ids{idA, idB, idB, idC, idC, idD, idD, idA, idC, idA};
        interface.setMeshEdges(meshID, 5, ids.data());
      } else {
        interface.setMeshEdge(meshID, idA, idB);
        interface.setMeshEdge(meshID, idB, idC);
        interface.setMeshEdge(meshID, idC, idD);
        interface.setMeshEdge(meshID, idD, idA);
        interface.setMeshEdge(meshID, idC, idA);
      }
    }

    if (useBulkFunctions) {
      std::vector ids{idA, idB, idC, idC, idD, idA};
      interface.setMeshTriangles(meshID, 2, ids.data());
    } else {
      interface.setMeshTriangle(meshID, idA, idB, idC);
      interface.setMeshTriangle(meshID, idC, idD, idA);
    }

    // Initialize, thus sending the mesh.
    double maxDt = interface.initialize();
    BOOST_TEST(interface.isCouplingOngoing(), "Sending participant should have to advance once!");

    // Write the data to be send.
    auto dataAID = "DataOne"; //  meshID
    BOOST_TEST(!interface.requiresGradientDataFor(meshID, dataAID));

    interface.writeScalarData(meshID, dataAID, idA, valOneA);
    interface.writeScalarData(meshID, dataAID, idB, valOneB);
    interface.writeScalarData(meshID, dataAID, idC, valOneC);
    interface.writeScalarData(meshID, dataAID, idD, valOneD);

    // Advance, thus send the data to the receiving partner.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Sending participant should have to advance once!");
    interface.finalize();
  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    precice::SolverInterface interface("SolverTwo", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshID = "MeshTwo";

    // Setup receiving mesh.
    int idA = interface.setMeshVertex(meshID, coordTwoA.data());
    int idB = interface.setMeshVertex(meshID, coordTwoB.data());
    int idC = interface.setMeshVertex(meshID, coordTwoC.data());

    // Initialize, thus receive the data and map.
    double maxDt = interface.initialize();
    BOOST_TEST(interface.isCouplingOngoing(), "Receiving participant should have to advance once!");

    // Read the mapped data from the mesh.
    auto dataAID = "DataOne"; //  meshID
    BOOST_TEST(!interface.requiresGradientDataFor(meshID, dataAID));

    double valueA, valueB, valueC;
    interface.readScalarData(meshID, dataAID, idA, valueA);
    interface.readScalarData(meshID, dataAID, idB, valueB);
    interface.readScalarData(meshID, dataAID, idC, valueC);

    BOOST_TEST(valueA == expectedValTwoA);
    BOOST_TEST(valueB == expectedValTwoB);
    BOOST_TEST(valueC == expectedValTwoC);

    // Verify that there is only one time step necessary.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Receiving participant should have to advance once!");
    interface.finalize();
  }
}

void testQuadMappingNearestProjection(bool defineEdgesExplicitly, bool useBulkFunctions, const std::string configFile, const TestContext &context)
{
  using Eigen::Vector3d;

  const double z = 0.3;

  // MeshOne
  Vector3d coordOneA{0.0, 0.0, z};
  Vector3d coordOneB{1.0, 0.0, z};
  Vector3d coordOneC{0.999999999, 1.0, z}; // Forces diagonal 0-2 to be shorter.
  Vector3d coordOneD{0.0, 1.0, z};
  double   valOneA = 1.0;
  double   valOneB = 3.0;
  double   valOneC = 5.0;
  double   valOneD = 7.0;

  // MeshTwo
  Vector3d coordTwoA{0.0, 0.0, z + 0.1};               // Maps to vertex A
  Vector3d coordTwoB{0.0, 0.5, z - 0.01};              // Maps to edge AD
  Vector3d coordTwoC{2.0 / 3.0, 1.0 / 3.0, z + 0.001}; // Maps to triangle ABC
  // This corresponds to the point C from mesh two on the triangle ABC on mesh one.
  Vector3d barycenterABC{0.3798734633239789, 0.24025307335204216, 0.3798734633239789};
  double   expectedValTwoA = 1.0;
  double   expectedValTwoB = 4.0;
  double   expectedValTwoC = Vector3d{valOneA, valOneB, valOneC}.dot(barycenterABC);

  if (context.isNamed("SolverOne")) {
    precice::SolverInterface interface("SolverOne", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshID = "MeshOne";

    // Setup mesh one.
    int idA = interface.setMeshVertex(meshID, coordOneA.data());
    int idB = interface.setMeshVertex(meshID, coordOneB.data());
    int idC = interface.setMeshVertex(meshID, coordOneC.data());
    int idD = interface.setMeshVertex(meshID, coordOneD.data());

    if (defineEdgesExplicitly) {
      if (useBulkFunctions) {
        std::vector ids{idA, idB, idB, idC, idC, idD, idD, idA};
        interface.setMeshEdges(meshID, 4, ids.data());
      } else {
        interface.setMeshEdge(meshID, idA, idB);
        interface.setMeshEdge(meshID, idB, idC);
        interface.setMeshEdge(meshID, idC, idD);
        interface.setMeshEdge(meshID, idD, idA);
      }
    }

    if (useBulkFunctions) {
      std::vector ids{idA, idB, idC, idD};
      interface.setMeshQuads(meshID, 1, ids.data());
    } else {
      interface.setMeshQuad(meshID, idA, idB, idC, idD);
    }

    auto &mesh = testing::WhiteboxAccessor::impl(interface).mesh("MeshOne");
    BOOST_REQUIRE(mesh.vertices().size() == 4);
    if (defineEdgesExplicitly) {
      BOOST_REQUIRE(mesh.edges().size() == 4);
    } else {
      BOOST_REQUIRE(mesh.edges().empty());
    }
    BOOST_REQUIRE(mesh.triangles().size() == 2);

    // Initialize, thus sending the mesh.
    double maxDt = interface.initialize();
    BOOST_TEST(mesh.edges().size() == 5);
    BOOST_TEST(mesh.triangles().size() == 2);

    BOOST_TEST(interface.isCouplingOngoing(), "Sending participant should have to advance once!");

    // Write the data to be send.
    auto dataAID = "DataOne"; //  meshID
    interface.writeScalarData(meshID, dataAID, idA, valOneA);
    interface.writeScalarData(meshID, dataAID, idB, valOneB);
    interface.writeScalarData(meshID, dataAID, idC, valOneC);
    interface.writeScalarData(meshID, dataAID, idD, valOneD);

    // Advance, thus send the data to the receiving partner.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Sending participant should have to advance once!");
    interface.finalize();
  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    precice::SolverInterface interface("SolverTwo", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshID = "MeshTwo";

    // Setup receiving mesh.
    int idA = interface.setMeshVertex(meshID, coordTwoA.data());
    int idB = interface.setMeshVertex(meshID, coordTwoB.data());
    int idC = interface.setMeshVertex(meshID, coordTwoC.data());

    // Initialize, thus receive the data and map.
    double maxDt = interface.initialize();
    BOOST_TEST(interface.isCouplingOngoing(), "Receiving participant should have to advance once!");

    // Read the mapped data from the mesh.
    auto   dataAID = "DataOne"; //  meshID
    double valueA, valueB, valueC;
    interface.readScalarData(meshID, dataAID, idA, valueA);
    interface.readScalarData(meshID, dataAID, idB, valueB);
    interface.readScalarData(meshID, dataAID, idC, valueC);

    BOOST_TEST(valueA == expectedValTwoA);
    BOOST_TEST(valueB == expectedValTwoB);
    BOOST_TEST(valueC == expectedValTwoC);

    // Verify that there is only one time step necessary.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Receiving participant should have to advance once!");
    interface.finalize();
  }
}

void testQuadMappingNearestProjectionTallKite(bool defineEdgesExplicitly, bool useBulkFunctions, const std::string configFile, const TestContext &context)
{
  using Eigen::Vector3d;

  const double z = 0.0;

  // MeshOne
  Vector3d coordOneA{-0.2, 0.0, z};
  Vector3d coordOneB{0.0, 0.5, z};
  Vector3d coordOneC{0.2, 0.0, z};
  Vector3d coordOneD{0.0, -0.5, z};

  if (context.isNamed("SolverOne")) {
    precice::SolverInterface interface("SolverOne", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshID = "MeshOne";

    // Setup mesh one.
    int idA = interface.setMeshVertex(meshID, coordOneA.data());
    int idB = interface.setMeshVertex(meshID, coordOneB.data());
    int idC = interface.setMeshVertex(meshID, coordOneC.data());
    int idD = interface.setMeshVertex(meshID, coordOneD.data());

    if (defineEdgesExplicitly) {
      if (useBulkFunctions) {
        std::vector ids{idA, idB, idB, idC, idC, idD, idD, idA};
        interface.setMeshEdges(meshID, 4, ids.data());
      } else {
        interface.setMeshEdge(meshID, idA, idB);
        interface.setMeshEdge(meshID, idB, idC);
        interface.setMeshEdge(meshID, idC, idD);
        interface.setMeshEdge(meshID, idD, idA);
      }
    }

    if (useBulkFunctions) {
      std::vector ids{idA, idB, idC, idD};
      interface.setMeshQuads(meshID, 1, ids.data());
    } else {
      interface.setMeshQuad(meshID, idA, idB, idC, idD);
    }

    auto &mesh = testing::WhiteboxAccessor::impl(interface).mesh("MeshOne");
    BOOST_REQUIRE(mesh.vertices().size() == 4);
    if (defineEdgesExplicitly) {
      BOOST_REQUIRE(mesh.edges().size() == 4);
    } else {
      BOOST_REQUIRE(mesh.edges().empty());
    }
    BOOST_REQUIRE(mesh.triangles().size() == 2);

    for (auto &edge : mesh.edges()) {
      BOOST_TEST(mesh::edgeLength(edge) < 0.6);
    }

    interface.finalize();
  }
}

void testQuadMappingNearestProjectionWideKite(bool defineEdgesExplicitly, bool useBulkFunctions, const std::string configFile, const TestContext &context)
{
  using Eigen::Vector3d;

  const double z = 0.0;

  // MeshOne
  Vector3d coordOneA{0.0, 0.0, z};
  Vector3d coordOneB{0.5, 0.2, z};
  Vector3d coordOneC{1.0, 0.0, z};
  Vector3d coordOneD{0.5, -0.2, z};

  if (context.isNamed("SolverOne")) {
    SolverInterface interface("SolverOne", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshID = "MeshOne";

    // Setup mesh one.
    int idA = interface.setMeshVertex(meshID, coordOneA.data());
    int idB = interface.setMeshVertex(meshID, coordOneB.data());
    int idC = interface.setMeshVertex(meshID, coordOneC.data());
    int idD = interface.setMeshVertex(meshID, coordOneD.data());

    if (defineEdgesExplicitly) {
      if (useBulkFunctions) {
        std::vector ids{idA, idB, idB, idC, idC, idD, idD, idA};
        interface.setMeshEdges(meshID, 4, ids.data());
      } else {
        interface.setMeshEdge(meshID, idA, idB);
        interface.setMeshEdge(meshID, idB, idC);
        interface.setMeshEdge(meshID, idC, idD);
        interface.setMeshEdge(meshID, idD, idA);
      }
    }

    if (useBulkFunctions) {
      std::vector ids{idA, idB, idD, idC};
      interface.setMeshQuads(meshID, 1, ids.data());
    } else {
      interface.setMeshQuad(meshID, idA, idB, idD, idC);
    }

    auto &mesh = testing::WhiteboxAccessor::impl(interface).mesh("MeshOne");
    BOOST_REQUIRE(mesh.vertices().size() == 4);
    if (defineEdgesExplicitly) {
      BOOST_REQUIRE(mesh.edges().size() == 4);
    } else {
      BOOST_REQUIRE(mesh.edges().empty());
    }
    BOOST_REQUIRE(mesh.triangles().size() == 2);

    for (auto &edge : mesh.edges()) {
      BOOST_TEST(mesh::edgeLength(edge) < 0.6);
    }

    interface.finalize();
  }
}

#endif
