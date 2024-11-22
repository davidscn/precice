#ifndef PRECICE_NO_MPI

#pragma once

#include "testing/TestContext.hpp"

using namespace precice;
using precice::testing::TestContext;

void testGreedyMapping(const std::string configFile, const TestContext &context, bool hasPolynomial, bool consistent);
void testGreedyMappingDirection2(const std::string configFile, const TestContext &context);
void testTimeDependentGreedyMapping(const std::string configFile, const TestContext &context);

#endif
