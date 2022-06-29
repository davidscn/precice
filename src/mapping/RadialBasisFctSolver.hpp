#pragma once

#include <Eigen/QR>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <numeric>
#include "mapping/config/MappingConfiguration.hpp"
#include "mapping/impl/BasisFunctions.hpp"
#include "mesh/Mesh.hpp"
#include "precice/types.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/Event.hpp"

namespace precice {
namespace mapping {

class RadialBasisFctSolver {
public:
  /// Default constructor
  RadialBasisFctSolver() = default;

  /// Assembles the system matrices and computes the decomposition of the interpolation matrix
  template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer>
  RadialBasisFctSolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                       const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial);

  /// Maps the given input data
  Eigen::VectorXd solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const;

  /// Maps the given input data
  Eigen::VectorXd solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const;

  // Clear all stored matrices
  void clear();

  // Access to the evaluation matrix
  const Eigen::MatrixXd &getEvaluationMatrix() const;

private:
  precice::logging::Logger _log{"mapping::RadialBasisFctSolver"};

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _qrMatrixC;

  // Decomposition of the polynomial
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _qrMatrixQ;

  // TODO: Add max columns and rows in the template parameter
  Eigen::MatrixXd _matrixQ;
  Eigen::MatrixXd _matrixV;

  Eigen::MatrixXd _matrixA;
};

// ------- Non-Member Functions ---------

/// Deletes all dead directions from fullVector and returns a vector of reduced dimensionality.
inline double computeSquaredDifference(
    const std::array<double, 3> &u,
    std::array<double, 3>        v,
    const std::array<bool, 3> &  activeAxis)
{
  // Substract the values and multiply out dead dimensions
  for (unsigned int d = 0; d < v.size(); ++d) {
    v[d] = (u[d] - v[d]) * static_cast<int>(activeAxis[d]);
  }
  // @todo: this can be replaced by std::hypot when moving to C++17
  return std::accumulate(v.begin(), v.end(), static_cast<double>(0.), [](auto &res, auto &val) { return res + val * val; });
}

// Fill in the polynomial entries
template <typename IndexContainer>
inline void fillPolynomialEntries(Eigen::MatrixXd &matrix, const mesh::Mesh &mesh, const IndexContainer &IDs, Eigen::Index startIndex, std::array<bool, 3> activeAxis)
{
  // Loop over all vertices in the mesh
  for (const auto &i : IDs | boost::adaptors::indexed()) {

    // 1. the constant contribution
    matrix(i.index(), startIndex) = 1.0;

    // 2. the linear contribution
    const auto & u = mesh.vertices()[i.value()].rawCoords();
    unsigned int k = 0;
    // Loop over all three space dimension and ignore dead axis
    for (unsigned int d = 0; d < activeAxis.size(); ++d) {
      if (activeAxis[d]) {
        PRECICE_ASSERT(matrix.rows() > i.index(), matrix.rows(), i.index());
        PRECICE_ASSERT(matrix.cols() > startIndex + 1 + k, matrix.cols(), startIndex + 1 + k);
        matrix(i.index(), startIndex + 1 + k) = u[d];
        ++k;
      }
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer>
Eigen::MatrixXd buildMatrixCLU(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                               std::array<bool, 3> activeAxis, Polynomial polynomial)
{
  // Treat the 2D case as 3D case with dead axis
  const unsigned int deadDimensions = std::count(activeAxis.begin(), activeAxis.end(), false);
  const unsigned int dimensions     = 3;
  const unsigned int polyparams     = polynomial == Polynomial::ON ? 1 + dimensions - deadDimensions : 0;

  // Add linear polynom degrees if polynomial requires this
  const auto inputSize = inputIDs.size();
  const auto n         = inputSize + polyparams;

  PRECICE_ASSERT((inputMesh.getDimensions() == 3) || activeAxis[2] == false);
  PRECICE_ASSERT((inputSize >= 1 + polyparams) || polynomial != Polynomial::ON, inputSize);

  Eigen::MatrixXd matrixCLU(n, n);
  matrixCLU.setZero();

  // Compute RBF matrix entries
  auto         i_iter  = inputIDs.begin();
  Eigen::Index i_index = 0;
  for (; i_iter != inputIDs.end(); ++i_iter, ++i_index) {
    const auto &u       = inputMesh.vertices()[*i_iter].rawCoords();
    auto        j_iter  = i_iter;
    auto        j_index = i_index;
    for (; j_iter != inputIDs.end(); ++j_iter, ++j_index) {
      const auto &v                 = inputMesh.vertices()[*j_iter].rawCoords();
      double      squaredDifference = computeSquaredDifference(u, v, activeAxis);
      matrixCLU(i_index, j_index)   = basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  // Add potentially the polynomial contribution in the matrix
  if (polynomial == Polynomial::ON) {
    fillPolynomialEntries(matrixCLU, inputMesh, inputIDs, inputSize, activeAxis);
  }
  matrixCLU.triangularView<Eigen::Lower>() = matrixCLU.transpose();
  return matrixCLU;
}

template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer>
Eigen::MatrixXd buildMatrixA(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                             const mesh::Mesh &outputMesh, const IndexContainer outputIDs, std::array<bool, 3> activeAxis, Polynomial polynomial)
{
  // Treat the 2D case as 3D case with dead axis
  const unsigned int deadDimensions = std::count(activeAxis.begin(), activeAxis.end(), false);
  const unsigned int dimensions     = 3;
  const unsigned int polyparams     = polynomial == Polynomial::ON ? 1 + dimensions - deadDimensions : 0;

  const auto inputSize  = inputIDs.size();
  const auto outputSize = outputIDs.size();
  const auto n          = inputSize + polyparams;

  PRECICE_ASSERT((inputMesh.getDimensions() == 3) || activeAxis[2] == false);
  PRECICE_ASSERT((inputSize >= 1 + polyparams) || polynomial != Polynomial::ON, inputSize);

  Eigen::MatrixXd matrixA(outputSize, n);
  matrixA.setZero();

  // Compute RBF values for matrix A
  for (const auto &i : outputIDs | boost::adaptors::indexed()) {
    const auto &u = outputMesh.vertices()[i.value()].rawCoords();
    for (const auto &j : inputIDs | boost::adaptors::indexed()) {
      const auto &v                 = inputMesh.vertices()[j.value()].rawCoords();
      double      squaredDifference = computeSquaredDifference(u, v, activeAxis);
      matrixA(i.index(), j.index()) = basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  // Add potentially the polynomial contribution in the matrix
  if (polynomial == Polynomial::ON) {
    fillPolynomialEntries(matrixA, outputMesh, outputIDs, inputSize, activeAxis);
  }

  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer>
RadialBasisFctSolver::RadialBasisFctSolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                           const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial)
{
  // Convert dead axis vector into an active axis array so that we can handle the reduction more easily
  std::array<bool, 3> activeAxis({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), activeAxis.begin(), [](const auto ax) { return !ax; });
  // First, assemble the interpolation matrix
  _qrMatrixC = buildMatrixCLU(basisFunction, inputMesh, inputIDs, activeAxis, polynomial).colPivHouseholderQr();

  PRECICE_CHECK(_qrMatrixC.isInvertible(),
                "The interpolation matrix of the RBF mapping from mesh {} to mesh {} is not invertable. "
                "This means that the mapping problem is not well-posed. "
                "Please check if your coupling meshes are correct. Maybe you need to fix axis-aligned mapping setups "
                "by marking perpendicular axes as dead?",
                inputMesh.getName(), outputMesh.getName());

  // Second, assemble evaluation matrix
  _matrixA = buildMatrixA(basisFunction, inputMesh, inputIDs, outputMesh, outputIDs, activeAxis, polynomial);

  // In case we deal with separated polynomials, we need dedicated matrices for the polynomial contribution
  if (polynomial == Polynomial::SEPARATE) {

    // 1. Allocate memory for these matrices
    // 4 = 1 + dimensions(3) = maximum number of polynomial parameters
    const unsigned int polyParams = 4 - std::count(activeAxis.begin(), activeAxis.end(), false);
    _matrixQ.resize(inputIDs.size(), polyParams);
    _matrixV.resize(outputIDs.size(), polyParams);

    // 2. fill the matrices: Q for the inputMesh, V for the outputMesh
    fillPolynomialEntries(_matrixQ, inputMesh, inputIDs, 0, activeAxis);
    fillPolynomialEntries(_matrixV, outputMesh, outputIDs, 0, activeAxis);

    // 3. compute decomposition
    _qrMatrixQ = _matrixQ.colPivHouseholderQr();
  }
}
} // namespace mapping
} // namespace precice
