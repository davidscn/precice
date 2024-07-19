#pragma once

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <numeric>
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"

namespace precice {
namespace mapping {

/**
 * VKOGA PGreedy algorithm
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class PGreedySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  PGreedySolver() = default;

  /**
   * assembles the system matrices and computes the decomposition of the interpolation matrix
   * inputMesh refers to the mesh where the interpolants are built on, i.e., the input mesh
   * for consistent mappings and the output mesh for conservative mappings
   * outputMesh refers to the mesh where we evaluate the interpolants, i.e., the output mesh
   * consistent mappings and the input mesh for conservative mappings
  */
  template <typename IndexContainer>
  PGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial);

  /// Maps the given input data
  Eigen::VectorXd solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const;

  /// Maps the given input data
  Eigen::VectorXd solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const;

  // Clear all stored matrices
  void clear();

  // Returns the size of the input data
  Eigen::Index getInputSize() const;

  // Returns the size of the input data
  Eigen::Index getOutputSize() const;

private:
  precice::logging::Logger _log{"mapping::PGreedySolver"};

  std::pair<mesh::Vertex, double> selectionRule(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction);
  Eigen::VectorXd                 predict(const mesh::Mesh::VertexContainer &vertices, RADIAL_BASIS_FUNCTION_T basisFunction);

  /// max iterations
  const int _max_iter = 10000;

  /// n_randon
  const double _tol_p = 1e-12;

  /// the selected centers
  mesh::Mesh::VertexContainer _centers;

  ///
  Eigen::MatrixXd _cut;

  /// Decomposition of the interpolation matrix
  DecompositionType _decMatrixC;

  /// Decomposition of the polynomial (for separate polynomial)
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _qrMatrixQ;

  /// Polynomial matrix of the input mesh (for separate polynomial)
  Eigen::MatrixXd _matrixQ;

  /// Polynomial matrix of the output mesh (for separate polynomial)
  Eigen::MatrixXd _matrixV;

  /// Evaluation matrix (output x input)
  Eigen::MatrixXd _matrixA;
};

// ------- Non-Member Functions ---------

/// Deletes all dead directions from fullVector and returns a vector of reduced dimensionality.
inline double computeSquaredDifference2(
    const std::array<double, 3> &u,
    std::array<double, 3>        v,
    const std::array<bool, 3> &  activeAxis = {{true, true, true}})
{
  // Subtract the values and multiply out dead dimensions
  for (unsigned int d = 0; d < v.size(); ++d) {
    v[d] = (u[d] - v[d]) * static_cast<int>(activeAxis[d]);
  }
  // @todo: this can be replaced by std::hypot when moving to C++17
  return std::accumulate(v.begin(), v.end(), static_cast<double>(0.), [](auto &res, auto &val) { return res + val * val; });
}

// template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer>
// Eigen::MatrixXd buildMatrixCLU(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
//                                std::array<bool, 3> activeAxis, Polynomial polynomial)
// {
//   // Treat the 2D case as 3D case with dead axis
//   const unsigned int deadDimensions = std::count(activeAxis.begin(), activeAxis.end(), false);
//   const unsigned int dimensions     = 3;
//   const unsigned int polyparams     = polynomial == Polynomial::ON ? 1 + dimensions - deadDimensions : 0;

//   // Add linear polynom degrees if polynomial requires this
//   const auto inputSize = inputIDs.size();
//   const auto n         = inputSize + polyparams;

//   PRECICE_ASSERT((inputMesh.getDimensions() == 3) || activeAxis[2] == false);
//   PRECICE_ASSERT((inputSize >= 1 + polyparams) || polynomial != Polynomial::ON, inputSize);

//   Eigen::MatrixXd matrixCLU(n, n);

//   // Compute RBF matrix entries
//   auto         i_iter  = inputIDs.begin();
//   Eigen::Index i_index = 0;
//   for (; i_iter != inputIDs.end(); ++i_iter, ++i_index) {
//     const auto &u       = inputMesh.vertex(*i_iter).rawCoords();
//     auto        j_iter  = i_iter;
//     auto        j_index = i_index;
//     for (; j_iter != inputIDs.end(); ++j_iter, ++j_index) {
//       const auto &v                 = inputMesh.vertex(*j_iter).rawCoords();
//       double      squaredDifference = computeSquaredDifference2(u, v, activeAxis);
//       matrixCLU(i_index, j_index)   = basisFunction.evaluate(std::sqrt(squaredDifference));
//     }
//   }

//   matrixCLU.triangularView<Eigen::Lower>() = matrixCLU.transpose();
//   return matrixCLU;
// }

template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer, typename VertexContainer>
Eigen::MatrixXd buildKernelMatrix(RADIAL_BASIS_FUNCTION_T basisFunction, const VertexContainer &inputMesh, const IndexContainer &inputIDs,
                                  VertexContainer &outputMesh, const IndexContainer outputIDs, std::array<bool, 3> activeAxis, Polynomial polynomial)
{
  std::cout << "Building kernel matrix:..." << std::endl;
  // Treat the 2D case as 3D case with dead axis
  const unsigned int deadDimensions = std::count(activeAxis.begin(), activeAxis.end(), false);
  const unsigned int dimensions     = 3;
  const unsigned int polyparams     = polynomial == Polynomial::ON ? 1 + dimensions - deadDimensions : 0;

  const auto inputSize  = inputIDs.size();
  const auto outputSize = outputIDs.size();
  const auto n          = inputSize + polyparams;

  // PRECICE_ASSERT((inputMesh.getDimensions() == 3) || activeAxis[2] == false);
  // PRECICE_ASSERT((inputSize >= 1 + polyparams) || polynomial != Polynomial::ON, inputSize);

  Eigen::MatrixXd matrixA(outputSize, n);

  // Compute RBF values for matrix A
  for (const auto &i : outputIDs | boost::adaptors::indexed()) {
    const auto &u = outputMesh.at(i.value()).rawCoords();
    for (const auto &j : inputIDs | boost::adaptors::indexed()) {
      const auto &v                 = inputMesh.at(j.value()).rawCoords();
      double      squaredDifference = computeSquaredDifference2(u, v, activeAxis);
      matrixA(i.index(), j.index()) = basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::predict(const mesh::Mesh::VertexContainer &vertices, RADIAL_BASIS_FUNCTION_T basisFunction)
{
  Eigen::VectorXd p(vertices.size());
  p.fill(basisFunction.evaluate(0));

  // First compute the diagonal entries
  // n = size of the centers
  if (!_centers.empty()) {
    auto n = _centers.size();
    // now compute (requires adjustment of the function) and only a portion of this matrix is required
    Eigen::MatrixXd kernel_eval = buildKernelMatrix(basisFunction, vertices, boost::irange<Eigen::Index>(0, n), _centers, boost::irange<Eigen::Index>(0, n), {{true, true, true}}, Polynomial::OFF);
    Eigen::VectorXd result      = (kernel_eval * _cut.block(0, 0, n, n).transpose()).array().square().rowwise().sum();
    p -= result;
  }
  //        # Otherwise check if everything is ok
  //        # Check is fit has been called
  //        check_is_fitted(self, 'ctrs_')
  //        # Validate the input
  //        X = check_array(X)
  //
  //        # Decide how many centers to use
  //        if n is None:
  //            n = np.atleast_2d(self.ctrs_).shape[0]
  //
  //        # Evaluate the power function on the input
  //  p = self.kernel.diagonal(X) - np.sum((self.kernel.eval(X, np.atleast_2d(self.ctrs_)[:n]) @ self.Cut_[:n, :n].transpose()) ** 2, axis=1)
  return p;
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<mesh::Vertex, double> PGreedySolver<RADIAL_BASIS_FUNCTION_T>::selectionRule(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction)
{
  // Sample is here just our input distribution
  Eigen::VectorXd p_X = predict(inputMesh.vertices(), basisFunction);
  Eigen::Index    maxIndex;
  double          maxValue = p_X.maxCoeff(&maxIndex);

  return {inputMesh.vertices().at(maxIndex), maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
PGreedySolver<RADIAL_BASIS_FUNCTION_T>::PGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial)
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  // Convert dead axis vector into an active axis array so that we can handle the reduction more easily
  std::array<bool, 3> activeAxis({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), activeAxis.begin(), [](const auto ax) { return !ax; });

  // Iterative selection of new points
  for (int n = 0; n < _max_iter; ++n) {

    // Select the current point
    std::cout << "Applying selection rule: ..." << std::endl;
    auto [x, pMax] = selectionRule(inputMesh, basisFunction);

    if (pMax < _tol_p) {
      break;
    }

    // Evaluate the first (n-1) bases on the selected point
    std::cout << "Computing Vx: ..." << std::endl;
    Eigen::MatrixXd Vx;
    if (n > 0) {
      Vx = buildKernelMatrix(basisFunction, inputMesh.vertices(), boost::irange<Eigen::Index>(0, inputMesh.vertices().size()), _centers, boost::irange<Eigen::Index>(0, n), {{true, true, true}}, Polynomial::OFF) *
           _cut.block(0, 0, n, n).transpose();
    }

    // Step 1: Append a column of zeros to the right of Cut_
    Eigen::MatrixXd cut_with_col                       = Eigen::MatrixXd::Zero(_cut.rows(), _cut.cols() + 1);
    cut_with_col.block(0, 0, _cut.rows(), _cut.cols()) = _cut;

    // Step 2: Append a row of zeros to the bottom of the resulting matrix
    Eigen::MatrixXd Cut_with_row_and_col                                       = Eigen::MatrixXd::Zero(_cut.rows() + 1, _cut.cols() + 1);
    Cut_with_row_and_col.block(0, 0, cut_with_col.rows(), cut_with_col.cols()) = cut_with_col;

    _cut = Cut_with_row_and_col;

    Eigen::RowVectorXd new_row = Eigen::RowVectorXd::Ones(n + 1);

    // Step 4: Update new_row if n > 0
    if (n > 0) {
      new_row.head(n) = (-Vx * _cut.block(0, 0, n, n)).row(0);
    }

    _cut.row(n) = new_row / std::sqrt(pMax);
    _centers.push_back(x);
    std::cout << "Number of centers: " << _centers.size() << std::endl;
  }
  // First, assemble the interpolation matrix and check the invertability
  // bool decompositionSuccessful = false;
  // if constexpr (RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite()) {
  //   _decMatrixC             = buildMatrixCLU(basisFunction, inputMesh, inputIDs, activeAxis, polynomial).llt();
  //   decompositionSuccessful = _decMatrixC.info() == Eigen::ComputationInfo::Success;
  // } else {
  //   _decMatrixC             = buildMatrixCLU(basisFunction, inputMesh, inputIDs, activeAxis, polynomial).colPivHouseholderQr();
  //   decompositionSuccessful = _decMatrixC.isInvertible();
  // }

  // PRECICE_CHECK(decompositionSuccessful,
  //               "The interpolation matrix of the RBF mapping from mesh \"{}\" to mesh \"{}\" is not invertable. "
  //               "This means that the mapping problem is not well-posed. "
  //               "Please check if your coupling meshes are correct (e.g. no vertices are duplicated) or reconfigure "
  //               "your basis-function (e.g. reduce the support-radius).",
  //               inputMesh.getName(), outputMesh.getName());

  // Second, assemble evaluation matrix
  // _matrixA = buildKernelMatrix(basisFunction, inputMesh, inputIDs, outputMesh, outputIDs, activeAxis, polynomial);

  // In case we deal with separated polynomials, we need dedicated matrices for the polynomial contribution
  // if (polynomial == Polynomial::SEPARATE) {

  //   // 4 = 1 + dimensions(3) = maximum number of polynomial parameters
  //   auto         localActiveAxis = activeAxis;
  //   unsigned int polyParams      = 4 - std::count(localActiveAxis.begin(), localActiveAxis.end(), false);

  //   do {
  //     // First, build matrix Q and check for the condition number
  //     _matrixQ.resize(inputIDs.size(), polyParams);
  //     fillPolynomialEntries(_matrixQ, inputMesh, inputIDs, 0, localActiveAxis);

  //     // Compute the condition number
  //     Eigen::JacobiSVD<Eigen::MatrixXd> svd(_matrixQ);
  //     PRECICE_ASSERT(svd.singularValues().size() > 0);
  //     PRECICE_DEBUG("Singular values in polynomial solver: {}", svd.singularValues());
  //     const double conditionNumber = svd.singularValues()(0) / std::max(svd.singularValues()(svd.singularValues().size() - 1), math::NUMERICAL_ZERO_DIFFERENCE);
  //     PRECICE_DEBUG("Condition number: {}", conditionNumber);

  //     // Disable one axis
  //     if (conditionNumber > 1e5) {
  //       reduceActiveAxis(inputMesh, inputIDs, localActiveAxis);
  //       polyParams = 4 - std::count(localActiveAxis.begin(), localActiveAxis.end(), false);
  //     } else {
  //       break;
  //     }
  //   } while (true);

  //   // allocate and fill matrix V for the outputMesh
  //   _matrixV.resize(outputIDs.size(), polyParams);
  //   fillPolynomialEntries(_matrixV, outputMesh, outputIDs, 0, localActiveAxis);

  //   // 3. compute decomposition
  //   _qrMatrixQ = _matrixQ.colPivHouseholderQr();
  // }
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  PRECICE_ASSERT((_matrixV.size() > 0 && polynomial == Polynomial::SEPARATE) || _matrixV.size() == 0, _matrixV.size());
  // TODO: Avoid temporary allocations
  // Au is equal to the eta in our PETSc implementation
  PRECICE_ASSERT(inputData.size() == _matrixA.rows());
  Eigen::VectorXd Au = _matrixA.transpose() * inputData;
  PRECICE_ASSERT(Au.size() == _matrixA.cols());

  // mu in the PETSc implementation
  Eigen::VectorXd out = _decMatrixC.solve(Au);

  if (polynomial == Polynomial::SEPARATE) {
    Eigen::VectorXd epsilon = _matrixV.transpose() * inputData;
    PRECICE_ASSERT(epsilon.size() == _matrixV.cols());

    // epsilon = Q^T * mu - epsilon (tau in the PETSc impl)
    epsilon -= _matrixQ.transpose() * out;
    PRECICE_ASSERT(epsilon.size() == _matrixQ.cols());

    // out  = out - solveTranspose tau (sigma in the PETSc impl)
    out -= static_cast<Eigen::VectorXd>(_qrMatrixQ.transpose().solve(-epsilon));
  }
  return out;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  PRECICE_ASSERT((_matrixQ.size() > 0 && polynomial == Polynomial::SEPARATE) || _matrixQ.size() == 0);
  Eigen::VectorXd polynomialContribution;
  // Solve polynomial QR and subtract it from the input data
  if (polynomial == Polynomial::SEPARATE) {
    polynomialContribution = _qrMatrixQ.solve(inputData);
    inputData -= (_matrixQ * polynomialContribution);
  }

  // Integrated polynomial (and separated)
  PRECICE_ASSERT(inputData.size() == _matrixA.cols());
  Eigen::VectorXd p = _decMatrixC.solve(inputData);
  PRECICE_ASSERT(p.size() == _matrixA.cols());
  Eigen::VectorXd out = _matrixA * p;

  // Add the polynomial part again for separated polynomial
  if (polynomial == Polynomial::SEPARATE) {
    out += (_matrixV * polynomialContribution);
  }
  return out;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedySolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _matrixA    = Eigen::MatrixXd();
  _decMatrixC = DecompositionType();
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _matrixA.cols();
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _matrixA.rows();
}
} // namespace mapping
} // namespace precice
