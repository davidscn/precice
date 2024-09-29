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
class FGreedySolver {
public:
  using DecompositionType = std::conditional_t<RADIAL_BASIS_FUNCTION_T::isStrictlyPositiveDefinite(), Eigen::LLT<Eigen::MatrixXd>, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>;
  using BASIS_FUNCTION_T  = RADIAL_BASIS_FUNCTION_T;
  /// Default constructor
  FGreedySolver() = default;

  /**
   * computes the greedy centers and stores data structures to later on evaluate the reduced model
  */
  template <typename IndexContainer>
  FGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
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
  precice::logging::Logger _log{"mapping::FGreedySolver"};

  std::pair<int, double> select(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction);
  Eigen::VectorXd        predict(const mesh::Mesh::VertexContainer &vertices, RADIAL_BASIS_FUNCTION_T basisFunction);

  /// max iterations
  const int _maxIter = 2000;

  /// n_randon
  const double _tolP = 1e-10;

  /// c upper triangular
  Eigen::MatrixXd _cut;
  Eigen::MatrixXd _resultV;

  std::vector<int> _greedyIDs;

  Eigen::Index    _inSize  = 0;
  Eigen::Index    _outSize = 0;
  Eigen::MatrixXd _kernel_eval;

  Eigen::MatrixXd _basisMatrix;
  Eigen::VectorXd _powerFunction;
};

// ------- Non-Member Functions ---------

/// Deletes all dead directions from fullVector and returns a vector of reduced dimensionality.
inline double computeSquaredDifference3(
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

template <typename RADIAL_BASIS_FUNCTION_T>
std::pair<int, double> FGreedySolver<RADIAL_BASIS_FUNCTION_T>::select(const mesh::Mesh &inputMesh, RADIAL_BASIS_FUNCTION_T basisFunction)
{
  // Sample is here just our input distribution
  Eigen::Index maxIndex;
  double       maxValue = _powerFunction.maxCoeff(&maxIndex);
  _powerFunction[maxIndex] = -std::numeric_limits<double>::infinity();
  return {maxIndex, maxValue};
}


template <typename RADIAL_BASIS_FUNCTION_T, typename VertexContainer>
Eigen::MatrixXd buildEvaluationMatrix2(RADIAL_BASIS_FUNCTION_T basisFunction, const VertexContainer &outputMesh, const VertexContainer &inputMesh, const std::vector<int> &greedyIDs)
{
  const mesh::Mesh::VertexContainer& inputVertices = inputMesh.vertices();
  const mesh::Mesh::VertexContainer& outputVertices = outputMesh.vertices();

  Eigen::MatrixXd matrixA(greedyIDs.size(), outputVertices.size());

  for (size_t i = 0; i < greedyIDs.size(); i++) 
  {
    const auto &u = inputVertices.at(greedyIDs.at(i)).rawCoords();
    for (size_t j = 0; j < outputVertices.size(); j++) 
    {
      const auto  &v                 = outputVertices.at(j).rawCoords();
      const double squaredDifference = computeSquaredDifference2(u, v, {{true, true, true}}); //TODO: Aktive Achsen
      matrixA(i, j)  = basisFunction.evaluate(std::sqrt(squaredDifference));
    }
  }

  return matrixA;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void updateKernelVector(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, std::vector<int> greedyIDs, Eigen::VectorXd &kernelVector, const mesh::Vertex &x)
{
  const mesh::Mesh::VertexContainer& vertices = inputMesh.vertices();
  for (size_t j = 0; j < greedyIDs.size(); j++)
  {
    const auto &y = vertices.at(greedyIDs.at(j)).rawCoords();
    kernelVector(j) = basisFunction.evaluate(std::sqrt(computeSquaredDifference2(x.rawCoords(), y, {{true, true, true}}))); //TODO: Aktive Achsen
  }
}

//TODO: SOLVER

template <typename RADIAL_BASIS_FUNCTION_T>
template <typename IndexContainer>
FGreedySolver<RADIAL_BASIS_FUNCTION_T>::FGreedySolver(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const IndexContainer &inputIDs,
                                                      const mesh::Mesh &outputMesh, const IndexContainer &outputIDs, std::vector<bool> deadAxis, Polynomial polynomial)
  : _inSize(inputMesh.vertices().size()), _outSize(outputMesh.vertices().size())
{
  PRECICE_ASSERT(polynomial == Polynomial::OFF, "Poly off");
  PRECICE_ASSERT(_greedyIDs.empty());
  PRECICE_ASSERT(_kernel_eval.size() == 0); 

  const int basisSize = std::min(static_cast<int>(_inSize), _maxIter); // maximal number of used basis functions
  _basisMatrix = Eigen::MatrixXd::Zero(_inSize, basisSize);
  _resultV = Eigen::MatrixXd::Zero(basisSize, basisSize);
  _cut = Eigen::MatrixXd::Zero(basisSize, basisSize);
  Eigen::VectorXd v(_inSize);
  _powerFunction = Eigen::VectorXd(_inSize); // Provisorisch
  _powerFunction.fill(basisFunction.evaluate(0)); // Provisorisch
  _greedyIDs.reserve(basisSize); //TODO: Aufräumen

  // Convert dead axis vector into an active axis array so that we can handle the reduction more easily
  std::array<bool, 3> activeAxis({{false, false, false}});
  std::transform(deadAxis.begin(), deadAxis.end(), activeAxis.begin(), [](const auto ax) { return !ax; });

  // Iterative selection of new points
  for (int n = 0; n < basisSize; ++n) {

    auto [i, pMax] = select(inputMesh, basisFunction);
    auto x = inputMesh.vertices().at(i); 

    if (pMax < _tolP) break;

    _greedyIDs.push_back(i);

    updateKernelVector(basisFunction, inputMesh, _greedyIDs, v, x);
    v.head(n) = _cut.block(0, 0, n, n) * v.head(n); // aliasing, alloziierung?
    v /= std::sqrt(v(n)); // v wird in Reihenfolge befüllt

    //_basisMatrix.col(n) = v;

    //_resultV.row(n) = _basisMatrix.row(i); // Teste Cholesky

    _cut.block(n, 0, 1, n).noalias() = -((Eigen::MatrixXd) v).block(0, 0, n, 1).transpose() * _cut.block(0, 0, n, n).triangularView<Eigen::Lower>();
    _cut(n,n) = 1;
    if (n > 0) _cut.block(n, 0, 1, n+1) /= v(n);
    std::cout << "iteration = " << n << "\r";
  }

  //mesh::Mesh centerMesh("greedy-centers", inputMesh.getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
  //centerMesh.vertices() = _centers;
  //io::ExportVTU exporter{"PGreedy", "exports", centerMesh, io::Export::ExportKind::TimeWindows, 1, /*Rank*/ 0, /*size*/ 1};
  //exporter.doExport(0, 0.0);

  _kernel_eval = buildEvaluationMatrix2(basisFunction, outputMesh, inputMesh, _greedyIDs);

  std::cout << "Mapping computed \n";
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConservative(const Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  // Not implemented
  PRECICE_ASSERT(false);
  return Eigen::VectorXd();
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::VectorXd FGreedySolver<RADIAL_BASIS_FUNCTION_T>::solveConsistent(Eigen::VectorXd &inputData, Polynomial polynomial) const
{
  Eigen::VectorXd y = inputData(_greedyIDs);

  //_basisMatrix(_greedyIDs, Eigen::all); geht nicht???

  Eigen::VectorXd beta       = _resultV.triangularView<Eigen::Lower>().solve(y);
  Eigen::VectorXd alpha      = _resultV.transpose().triangularView<Eigen::Upper>().solve(beta);
  Eigen::VectorXd prediction = _kernel_eval.transpose() * alpha;

  return prediction;
}


template <typename RADIAL_BASIS_FUNCTION_T>
void FGreedySolver<RADIAL_BASIS_FUNCTION_T>::clear() //TODO: clear() anpassen
{
  _greedyIDs.clear();
  _kernel_eval = Eigen::MatrixXd();
  _inSize      = 0;
  _outSize     = 0;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedySolver<RADIAL_BASIS_FUNCTION_T>::getInputSize() const
{
  return _inSize;
}


template <typename RADIAL_BASIS_FUNCTION_T>
Eigen::Index FGreedySolver<RADIAL_BASIS_FUNCTION_T>::getOutputSize() const
{
  return _outSize;
}
} // namespace mapping
} // namespace precice
