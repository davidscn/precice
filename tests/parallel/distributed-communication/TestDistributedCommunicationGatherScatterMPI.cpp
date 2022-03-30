#ifndef PRECICE_NO_MPI

#include "testing/Testing.hpp"

#include <precice/SolverInterface.hpp>
#include "helpers.hpp"

BOOST_AUTO_TEST_SUITE(PreciceTests)
BOOST_AUTO_TEST_SUITE(Parallel)
BOOST_AUTO_TEST_SUITE(DistributedCommunication)
BOOST_AUTO_TEST_CASE(TestDistributedCommunicationGatherScatterMPI)
{
  PRECICE_TEST("Fluid"_on(2_ranks), "Structure"_on(2_ranks));
  runTestDistributedCommunication(context.config(), context);
}

BOOST_AUTO_TEST_SUITE_END() // PreciceTests
BOOST_AUTO_TEST_SUITE_END() // Parallel
BOOST_AUTO_TEST_SUITE_END() // DistributedCommunication

#endif // PRECICE_NO_MPI