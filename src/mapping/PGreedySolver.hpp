#pragma once

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/irange.hpp>
#include <numeric>
#include "io/ExportVTU.hpp"
#include "mapping/config/MappingConfigurationTypes.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
namespace precice {
namespace mapping {

/**
 * VKOGA PGreedy algorithm: reimplemented the PGreedy solver as found in
 * https://github.com/GabrieleSantin/VKOGA/blob/master/src/vkoga/pgreedy.py
 *
 * As opposed to the original example in VKOGA, our setup is slightly different in terms of when to compute what:
 *
 * Nomenclature:
 * Original: X -> our case: input mesh vertices, i.e.,
 * the vertices in two or three dimensional space on which we have data given and on which we want to build an interpolant
 * Original: Y -> our case: input data, i.e., the coupling data associated to the input mesh vertices
 * Original: X_test -> our case: output mesh vertices, i.e., the vertices (2d or 3d) on which we need to evaluate the interpolant
 * Original: Y_test -> our case: output mesh data, i.e., the unknown data values we want to evaluate, associated to the output mesh vertices
 *
 * In the original case, we typically have initially given: X and Y, such that we have two main stages:
 *
 * 1. PGreedy(params) and PGreedy.fit(X, y), which creates the reduced model
 * 2. PGreedy.predict(X_test), which evaluates the fit on the test data
 *
 * In our case, we typically have initially given: X and X_test, such that we have two (different) main stages:
 *
 * 1. PGreedy(params, X, X_test), which computes the centers and associated data structures (_cut and greedyIDs)
 * 2. PGreedy.solveConsistent(y), which evaluates the model for new data
 *
 * When an object is created, we compute the centers, the solveConsistent evaluates the center fit for new data.
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class PGreedySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  PGreedySolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
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

  std::pair<int, double> selectionRule(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction);
  Eigen::VectorXd        predict(const mesh::Mesh::VertexContainer &vertices, RADIAL_BASIS_FUNCTION_T basisFunction);

  /// max iterations
  const int _max_iter = 100;

  /// n_randon
  const double _tol_p = 1e-10;

  /// the selected centers
  mesh::Mesh::VertexContainer _centers;

  /// c upper triangular
  Eigen::MatrixXd _cut;

  std::vector<int> _greedyIDs;

  Eigen::Index    _inSize  = 0;
  Eigen::Index    _outSize = 0;
  Eigen::MatrixXd _kernel_eval;
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

template <typename RADIAL_BASIS_FUNCTION_T, typename IndexContainer, typename VertexContainer>
Eigen::MatrixXd buildKernelMatrix(RADIAL_BASIS_FUNCTION_T basisFunction, const VertexContainer &inputMesh, const IndexContainer &inputIDs,
                                  VertexContainer &outputMesh, const IndexContainer outputIDs, std::array<bool, 3> activeAxis, Polynomial polynomial)
{
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
    Eigen::MatrixXd kernel_eval = buildKernelMatrix(basisFunction, vertices, boost::irange<Eigen::Index>(0, vertices.size()), _centers, boost::irange<Eigen::Index>(0, n), {{true, true, true}}, Polynomial::OFF);
    Eigen::VectorXd result      = (kernel_eval.transpose() * _cut.block(0, 0, n, n).transpose()).array().square().rowwise().sum();
    p -= result;
  }
  return p;
}

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> PGreedySolver<RADIAL_BASIS_FUNCTION_T>::selectionRule(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction)
{
  // Sample is here just our input distribution
  Eigen::VectorXd p_X = predict(inputMesh.vertices(), basisFunction);
  Eigen::Index    maxIndex;
  double          maxValue = p_X.maxCoeff(&maxIndex);

  return {maxIndex, maxValue};
}

template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
PGreedySolver<RADIAL_BASIS_FUNCTION_T>::PGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial)
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(_centers.empty());
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_cut.size() == 0);
  PRECICE_ASSERT(_kernel_eval.size() == 0);

  _inSize  = inputMesh.vertices().size();
  _outSize = outputMesh.vertices().size();

  // Convert dead axis vector into an active axis array so that we can handle the reduction more easily
  std::array<bool, 3> activeAxis({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), activeAxis.begin(), [](const auto ax) { return !ax; });

  // Iterative selection of new points
  for (int n = 0; n < _max_iter; ++n) {
    // Select the current point
    auto [ind, pMax] = selectionRule(inputMesh, basisFunction);

    auto x = inputMesh.vertices().at(ind);

    std::cout << "Iteration: " << n << std::endl;
    std::cout << "pMax: " << pMax << std::endl;
    if (pMax < _tol_p) {
      break;
    }

    // Evaluate the first (n-1) bases on the selected point
    Eigen::MatrixXd Vx;
    if (n > 0) {
      Vx = buildKernelMatrix(basisFunction, mesh::Mesh::VertexContainer{x}, boost::irange<Eigen::Index>(0, 1), _centers, boost::irange<Eigen::Index>(0, n), {{true, true, true}}, Polynomial::OFF).transpose() *
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
    _greedyIDs.push_back(ind);
    _centers.push_back(x);
  }

  // If the mesh creation is shifted into the loop (centerMesh.vertices()), one could visualize the distribution of pMax
  mesh::Mesh centerMesh("greedy-centers", inputMesh.getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
  centerMesh.vertices() = _centers;
  io::ExportVTU exporter{"PGreedy", "exports", centerMesh, io::Export::ExportKind::TimeWindows, 1, /*Rank*/ 0, /*size*/ 1};
  exporter.doExport(0, 0.0);

  // now compute (requires adjustment of the function) and only a portion of this matrix is required
  _kernel_eval = buildKernelMatrix(basisFunction, outputMesh.vertices(), boost::irange<Eigen::Index>(0, outputMesh.vertices().size()), _centers, boost::irange<Eigen::Index>(0, _centers.size()), {{true, true, true}}, Polynomial::OFF);
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd PGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const
{

  // First, compute the c vector
  // see https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html
  Eigen::VectorXd c = inputData(_greedyIDs);

  Eigen::VectorXd coeff      = _cut.triangularView<Eigen::Lower>().transpose() * _cut * c;
  Eigen::VectorXd prediction = _kernel_eval.transpose() * coeff;

  return prediction;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void PGreedySolver<RADIAL_BASIS_FUNCTION_T>::clear()
{
  _centers.clear();
  _greedyIDs.clear();
  _cut         = Eigen::MatrixXd();
  _kernel_eval = Eigen::MatrixXd();
  _inSize      = 0;
  _outSize     = 0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}

template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index PGreedySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice
