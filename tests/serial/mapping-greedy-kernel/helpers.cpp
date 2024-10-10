#ifndef PRECICE_NO_MPI

#include "helpers.hpp"
#include "testing/Testing.hpp"

#include "mesh/Utils.hpp"
#include "precice/impl/ParticipantImpl.hpp"
#include "precice/precice.hpp"

std::vector<int> generateMeshOne(precice::Participant &interface, const std::string &meshOneID)
{
  const double     z = 0.3;
  std::vector<int> ids;
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{0.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{1.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{1.0, 1.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{0.0, 1.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{2.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{3.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{3.0, 1.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{2.0, 1.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{4.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{5.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{5.0, 1.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshOneID, Eigen::Vector3d{4.0, 1.0, z}));
  return ids;
}

std::vector<int> generateMeshTwo(precice::Participant &interface, const std::string &meshTwoID)
{
  const double     z = 0.3;
  std::vector<int> ids;
  ids.emplace_back(interface.setMeshVertex(meshTwoID, Eigen::Vector3d{0.0, 0.0, z}));
  ids.emplace_back(interface.setMeshVertex(meshTwoID, Eigen::Vector3d{0.5, 0.5, z}));
  ids.emplace_back(interface.setMeshVertex(meshTwoID, Eigen::Vector3d{3.5, 0.5, z}));
  return ids;
}

void testGreedyMappingDirection1(const std::string configFile, const TestContext &context)
{
  using Eigen::Vector3d;

  std::vector<double> values;
  for (unsigned int i = 0; i < 12; ++i)
    values.emplace_back(std::pow(i + 1, 2));

  double expectedValues[3] = {1.0000000000000002, 2.1879131472090689, 22.859664317930537};

  if (context.isNamed("SolverOne")) {
    precice::Participant interface("SolverOne", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshOneID = "MeshOne";

    // Setup mesh one.
    std::vector<int> ids = generateMeshOne(interface, meshOneID);

    // Initialize, thus sending the mesh.
    interface.initialize();
    double maxDt = interface.getMaxTimeStepSize();
    BOOST_TEST(interface.isCouplingOngoing(), "Sending participant should have to advance once!");

    // Write the data to be send.
    auto dataAID = "DataOne";
    BOOST_TEST(!interface.requiresGradientDataFor(meshOneID, dataAID));
    interface.writeData(meshOneID, dataAID, ids, values);

    // Advance, thus send the data to the receiving partner.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Sending participant should have to advance once!");
    interface.finalize();

  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    precice::Participant interface("SolverTwo", configFile, 0, 1);
    // namespace is required because we are outside the fixture
    auto meshTwoID = "MeshTwo";

    // Setup receiving mesh.
    std::vector<int> ids = generateMeshTwo(interface, meshTwoID);

    // Initialize, thus receive the data and map.
    interface.initialize();
    double maxDt = interface.getMaxTimeStepSize();
    BOOST_TEST(interface.isCouplingOngoing(), "Receiving participant should have to advance once!");

    // Read the mapped data from the mesh.
    auto dataAID = "DataOne";
    BOOST_TEST(!interface.requiresGradientDataFor(meshTwoID, dataAID));

    double values[3];
    interface.readData(meshTwoID, dataAID, ids, maxDt, values);

    for (int i = 0; i < 3; i++) {
      BOOST_TEST(values[i] == expectedValues[i], boost::test_tools::tolerance(1e-7));
    }

    // Verify that there is only one time step necessary.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Receiving participant should have to advance once!");
    interface.finalize();
  }
}

void testGreedyMappingDirection2(const std::string configFile, const TestContext &context)
{
  using Eigen::Vector3d;

  auto meshOneID = "MeshOne";
  auto meshTwoID = "MeshTwo";
  auto dataAID   = "DataOne";

  std::vector<double> values;
  for (unsigned int i = 0; i < 3; ++i)
    values.emplace_back(std::pow(i + 1, 2));

  double expectedValues[12] = {
      1,
      0.29995768373298215,
      0.296021056494793,
      0.29995768373298215,
      3.070337108988407e-05,
      0.6749047883992098,
      0.6749047883992098,
      3.070266708867809e-05,
      0.6749047883992098,
      2.1342363350954833e-05,
      2.1342363350954833e-05,
      0.6749047883992098};

  if (context.isNamed("SolverOne")) {
    precice::Participant interface("SolverOne", configFile, 0, 1);

    std::vector<int> idsMeshOne = generateMeshTwo(interface, meshOneID);

    // Initialize, thus sending the mesh.
    interface.initialize();
    double maxDt = interface.getMaxTimeStepSize();
    BOOST_TEST(interface.isCouplingOngoing(), "Sending participant should have to advance once!");

    // Write the data to be send.
    BOOST_TEST(!interface.requiresGradientDataFor(meshOneID, dataAID));
    interface.writeData(meshOneID, dataAID, idsMeshOne, values);

    // Advance, thus send the data to the receiving partner.
    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Sending participant should have to advance once!");
    interface.finalize();

  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    precice::Participant interface("SolverTwo", configFile, 0, 1);

    std::vector<int> idsMeshTwo = generateMeshOne(interface, meshTwoID);

    // Initialize, thus receive the data and map.
    interface.initialize();
    double maxDt = interface.getMaxTimeStepSize();
    BOOST_TEST(interface.isCouplingOngoing(), "Receiving participant should have to advance once!");

    // Read the mapped data from the mesh.
    BOOST_TEST(!interface.requiresGradientDataFor(meshTwoID, dataAID));

    double values[12];
    interface.readData(meshTwoID, dataAID, idsMeshTwo, maxDt, values);

    // Due to Eigen 3.3.7 (Ubunu 2004) giving slightly different results
    for (int i = 0; i < 12; i++) {
      BOOST_TEST(values[i] == expectedValues[i], boost::test_tools::tolerance(1e-7));
    }

    interface.advance(maxDt);
    BOOST_TEST(!interface.isCouplingOngoing(), "Receiving participant should have to advance once!");
    interface.finalize();
  }
}

#endif
