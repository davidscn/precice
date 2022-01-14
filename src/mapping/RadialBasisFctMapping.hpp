#pragma once

#include <Eigen/Core>
#include <Eigen/QR>

#include "com/CommunicateMesh.hpp"
#include "com/Communication.hpp"
#include "config/MappingConfiguration.hpp"
#include "impl/BasisFunctions.hpp"
#include "mapping/Mapping.hpp"
#include "mesh/Filter.hpp"
#include "precice/types.hpp"
#include "query/Index.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/Event.hpp"
#include "utils/MasterSlave.hpp"

namespace precice {
extern bool syncMode;

namespace mapping {

/**
 * @brief Mapping with radial basis functions.
 *
 * With help of the input data points and values an interpolant is constructed.
 * The interpolant is formed by a weighted sum of conditionally positive radial
 * basis functions and a (low order) polynomial, and evaluated at the output
 * data points.
 *
 * The radial basis function type has to be given as template parameter, and has
 * to be one of the defined types in this file.
 */
template <typename RADIAL_BASIS_FUNCTION_T>
class RadialBasisFctMapping : public Mapping {
public:
  /**
   * @brief Constructor.
   *
   * @param[in] constraint Specifies mapping to be consistent or conservative.
   * @param[in] dimensions Dimensionality of the meshes
   * @param[in] function Radial basis function used for mapping.
   * @param[in] xDead, yDead, zDead Deactivates mapping along an axis
   */
  RadialBasisFctMapping(
      Constraint              constraint,
      int                     dimensions,
      RADIAL_BASIS_FUNCTION_T function,
      bool                    xDead,
      bool                    yDead,
      bool                    zDead,
      Polynomial              polynomial = Polynomial::SEPARATE);

  /// Computes the mapping coefficients from the in- and output mesh.
  virtual void computeMapping() override;

  /// Returns true, if computeMapping() has been called.
  virtual bool hasComputedMapping() const override;

  /// Removes a computed mapping.
  virtual void clear() override;

  /// Maps input data to output data from input mesh to output mesh.
  virtual void map(int inputDataID, int outputDataID) override;

  virtual void tagMeshFirstRound() override;

  virtual void tagMeshSecondRound() override;

private:
  precice::logging::Logger _log{"mapping::RadialBasisFctMapping"};

  bool _hasComputedMapping = false;

  /// Radial basis function type used in interpolation.
  RADIAL_BASIS_FUNCTION_T _basisFunction;

  /// Interpolation evaluation matrix. Evaluated basis function on the output mesh
  Eigen::MatrixXd _matrixA;

  /// QR decomposed system matrix. Evaluated basis function on the input mesh
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _qrMatrixC;

  /// Vandermonde Matrix for linear polynomial, constructed from vertices of the input mesh
  // Eigen::MatrixXd _matrixQ;

  /// Coordinates of the output mesh to evaluate the separated polynomial
  // Eigen::MatrixXd _matrixV;

  /// true if the mapping along some axis should be ignored
  std::vector<bool> _deadAxis;

  /// Toggles the use of the additonal polynomial
  Polynomial _polynomial;

  /// Number of coefficients for the integrated polynomial. Depends on dimension and number of dead dimensions
  unsigned int polyparams;

  /// Number of coefficients for the separated polynomial. Depends on dimension and number of dead dimensions
  unsigned int sepPolyparams;

  void mapConservative(int inputDataID, int outputDataID);
  void mapConsistent(int inputDataID, int outputDataID);

  // Set dead axis in the _deadAxis vector
  void setDeadAxis(bool xDead, bool yDead, bool zDead);
};

// --------------------------------------------------- HEADER IMPLEMENTATIONS

template <typename RADIAL_BASIS_FUNCTION_T>
RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::RadialBasisFctMapping(
    Constraint              constraint,
    int                     dimensions,
    RADIAL_BASIS_FUNCTION_T function,
    bool                    xDead,
    bool                    yDead,
    bool                    zDead,
    Polynomial              polynomial)
    : Mapping(constraint, dimensions),
      _basisFunction(function),
      _polynomial(polynomial)

{
  if (constraint == SCALEDCONSISTENT) {
    setInputRequirement(Mapping::MeshRequirement::FULL);
    setOutputRequirement(Mapping::MeshRequirement::FULL);
  } else {
    setInputRequirement(Mapping::MeshRequirement::VERTEX);
    setOutputRequirement(Mapping::MeshRequirement::VERTEX);
  }

  PRECICE_CHECK(polynomial == Polynomial::ON || polynomial == Polynomial::SEPARATE, "RBF mappings without a polynomial ( polynomial = \"off\") are only implemented using PETSc. Please install preCICE with PETSc or remove the use-qr-decomposition=\"true\" tag in your configuration file.");
  setDeadAxis(xDead, yDead, zDead);

  // Count number of dead dimensions
  int deadDimensions = 0;
  for (int d = 0; d < dimensions; d++) {
    if (_deadAxis[d])
      deadDimensions += 1;
  }
  polyparams    = (_polynomial == Polynomial::ON) ? 1 + dimensions - deadDimensions : 0;
  sepPolyparams = (_polynomial == Polynomial::SEPARATE) ? 1 + dimensions - deadDimensions : 0;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::setDeadAxis(bool xDead, bool yDead, bool zDead)
{
  // TODO: Remove duplication here with PETSc based implementation
  _deadAxis.resize(getDimensions());
  if (getDimensions() == 2) {
    _deadAxis = {xDead, yDead};
    PRECICE_CHECK(not(xDead and yDead),
                  "You cannot set all axes to dead for an RBF mapping. "
                  "Please remove one of the respective mapping's \"x-dead\" or \"y-dead\" attributes.");
    if (zDead)
      PRECICE_WARN("Setting the z-axis to dead on a 2-dimensional problem has no effect. "
                   "Please remove the respective mapping's \"z-dead\" attribute.");
  } else if (getDimensions() == 3) {
    _deadAxis = {xDead, yDead, zDead};
    PRECICE_CHECK(not(xDead and yDead and zDead),
                  "You cannot set all axes to dead for an RBF mapping. "
                  "Please remove one of the respective mapping's \"x-dead\", \"y-dead\", or \"z-dead\" attributes.");
  } else {
    PRECICE_ASSERT(false);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping()
{
  PRECICE_TRACE();

  precice::utils::Event e("map.rbf.computeMapping.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  PRECICE_ASSERT(input()->getDimensions() == output()->getDimensions(),
                 input()->getDimensions(), output()->getDimensions());
  PRECICE_ASSERT(getDimensions() == output()->getDimensions(),
                 getDimensions(), output()->getDimensions());

  // Some Debug information
  if (_polynomial == Polynomial::ON) {
    PRECICE_DEBUG("Using integrated polynomial.");
  }
  if (_polynomial == Polynomial::SEPARATE) {
    PRECICE_DEBUG("Using seperated polynomial.");
  }

  // Determine input and output
  mesh::PtrMesh inMesh;
  mesh::PtrMesh outMesh;
  if (hasConstraint(CONSERVATIVE)) {
    inMesh  = output();
    outMesh = input();
  } else { // Consistent or scaled consistent
    inMesh  = input();
    outMesh = output();
  }

  // Handle the gather-scatter of the meshes
  if (utils::MasterSlave::isSlave()) {

    // Input mesh may have overlaps
    mesh::Mesh filteredInMesh("filteredInMesh", inMesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
    mesh::filterMesh(filteredInMesh, *inMesh, [&](const mesh::Vertex &v) { return v.isOwner(); });

    // Send the mesh
    com::CommunicateMesh(utils::MasterSlave::_communication).sendMesh(filteredInMesh, 0);
    com::CommunicateMesh(utils::MasterSlave::_communication).sendMesh(*outMesh, 0);

  } else { // Parallel Master or Serial

    mesh::Mesh globalInMesh("globalInMesh", inMesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
    mesh::Mesh globalOutMesh("globalOutMesh", outMesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);

    if (utils::MasterSlave::isMaster()) {
      {
        // Input mesh may have overlaps
        mesh::Mesh filteredInMesh("filteredInMesh", inMesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
        mesh::filterMesh(filteredInMesh, *inMesh, [&](const mesh::Vertex &v) { return v.isOwner(); });
        globalInMesh.addMesh(filteredInMesh);
        globalOutMesh.addMesh(*outMesh);
      }

      // Receive mesh
      for (Rank rankSlave : utils::MasterSlave::allSlaves()) {
        mesh::Mesh slaveInMesh(inMesh->getName(), inMesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
        com::CommunicateMesh(utils::MasterSlave::_communication).receiveMesh(slaveInMesh, rankSlave);
        globalInMesh.addMesh(slaveInMesh);

        mesh::Mesh slaveOutMesh(outMesh->getName(), outMesh->getDimensions(), mesh::Mesh::MESH_ID_UNDEFINED);
        com::CommunicateMesh(utils::MasterSlave::_communication).receiveMesh(slaveOutMesh, rankSlave);
        globalOutMesh.addMesh(slaveOutMesh);
      }

    } else { // Serial
      globalInMesh.addMesh(*inMesh);
      globalOutMesh.addMesh(*outMesh);
    }

    _matrixA   = buildMatrixA(_basisFunction, globalInMesh, globalOutMesh, _deadAxis);
    _qrMatrixC = buildMatrixCLU(_basisFunction, globalInMesh, _deadAxis).colPivHouseholderQr();

    PRECICE_CHECK(_qrMatrixC.isInvertible(),
                  "The interpolation matrix of the RBF mapping from mesh {} to mesh {} is not invertable. "
                  "This means that the mapping problem is not well-posed. "
                  "Please check if your coupling meshes are correct. Maybe you need to fix axis-aligned mapping setups "
                  "by marking perpendicular axes as dead?",
                  input()->getName(), output()->getName());
  }
  _hasComputedMapping = true;
  PRECICE_DEBUG("Compute Mapping is Completed.");
} // namespace mapping

template <typename RADIAL_BASIS_FUNCTION_T>
bool RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::hasComputedMapping() const
{
  return _hasComputedMapping;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::clear()
{
  PRECICE_TRACE();
  _matrixA            = Eigen::MatrixXd();
  _qrMatrixC          = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>();
  _hasComputedMapping = false;
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::map(
    int inputDataID,
    int outputDataID)
{
  PRECICE_TRACE(inputDataID, outputDataID);

  precice::utils::Event e("map.rbf.mapData.From" + input()->getName() + "To" + output()->getName(), precice::syncMode);

  PRECICE_ASSERT(_hasComputedMapping);
  PRECICE_ASSERT(input()->getDimensions() == output()->getDimensions(),
                 input()->getDimensions(), output()->getDimensions());
  PRECICE_ASSERT(getDimensions() == output()->getDimensions(),
                 getDimensions(), output()->getDimensions());
  {
    int valueDim = input()->data(inputDataID)->getDimensions();
    PRECICE_ASSERT(valueDim == output()->data(outputDataID)->getDimensions(),
                   valueDim, output()->data(outputDataID)->getDimensions());
  }

  if (hasConstraint(CONSERVATIVE)) {
    mapConservative(inputDataID, outputDataID);
  } else {
    mapConsistent(inputDataID, outputDataID);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::mapConservative(int inputDataID, int outputDataID)
{

  PRECICE_TRACE(inputDataID, outputDataID, polyparams);

  // Gather input data
  if (utils::MasterSlave::isSlave()) {

    const auto &localInData = input()->data(inputDataID)->values();

    int localOutputSize = 0;
    for (const auto &vertex : output()->vertices()) {
      if (vertex.isOwner()) {
        ++localOutputSize;
      }
    }

    localOutputSize *= output()->data(outputDataID)->getDimensions();

    utils::MasterSlave::_communication->send(localInData, 0);
    utils::MasterSlave::_communication->send(localOutputSize, 0);

  } else { // Parallel Master or Serial case

    std::vector<double> globalInValues;
    std::vector<double> outputValueSizes;
    {
      const auto &localInData = input()->data(inputDataID)->values();
      globalInValues.insert(globalInValues.begin(), localInData.data(), localInData.data() + localInData.size());

      int localOutputSize = 0;
      for (const auto &vertex : output()->vertices()) {
        if (vertex.isOwner()) {
          ++localOutputSize;
        }
      }

      localOutputSize *= output()->data(outputDataID)->getDimensions();

      outputValueSizes.push_back(localOutputSize);
    }

    {
      std::vector<double> slaveBuffer;
      int                 slaveOutputValueSize;
      for (Rank rank : utils::MasterSlave::allSlaves()) {
        utils::MasterSlave::_communication->receive(slaveBuffer, rank);
        globalInValues.insert(globalInValues.end(), slaveBuffer.begin(), slaveBuffer.end());

        utils::MasterSlave::_communication->receive(slaveOutputValueSize, rank);
        outputValueSizes.push_back(slaveOutputValueSize);
      }
    }

    int valueDim = output()->data(outputDataID)->getDimensions();

    // Construct Eigen vectors
    Eigen::Map<Eigen::VectorXd> inputValues(globalInValues.data(), globalInValues.size());
    Eigen::VectorXd             outputValues((_matrixA.cols() - polyparams) * valueDim);
    outputValues.setZero();

    Eigen::VectorXd Au(_matrixA.cols());  // rows == n
    Eigen::VectorXd in(_matrixA.rows());  // rows == outputSize
    Eigen::VectorXd out(_matrixA.cols()); // rows == n

    for (int dim = 0; dim < valueDim; dim++) {
      for (int i = 0; i < in.size(); i++) { // Fill input data values
        in[i] = inputValues(i * valueDim + dim);
      }

      Au  = _matrixA.transpose() * in;
      out = _qrMatrixC.solve(Au);

      // Copy mapped data to output data values
      for (int i = 0; i < out.size() - polyparams; i++) {
        outputValues[i * valueDim + dim] = out[i];
      }
    }

    // Data scattering to slaves
    if (utils::MasterSlave::isMaster()) {

      // Filter data
      int outputCounter = 0;
      for (int i = 0; i < static_cast<int>(output()->vertices().size()); ++i) {
        if (output()->vertices()[i].isOwner()) {
          for (int dim = 0; dim < valueDim; ++dim) {
            output()->data(outputDataID)->values()[i * valueDim + dim] = outputValues(outputCounter);
            ++outputCounter;
          }
        }
      }

      // Data scattering to slaves
      int beginPoint = outputValueSizes.at(0);
      for (Rank rank : utils::MasterSlave::allSlaves()) {
        precice::span<const double> toSend{outputValues.data() + beginPoint, static_cast<size_t>(outputValueSizes.at(rank))};
        utils::MasterSlave::_communication->send(toSend, rank);
        beginPoint += outputValueSizes.at(rank);
      }
    } else { // Serial
      output()->data(outputDataID)->values() = outputValues;
    }
  }
  if (utils::MasterSlave::isSlave()) {
    std::vector<double> receivedValues;
    utils::MasterSlave::_communication->receive(receivedValues, 0);

    int valueDim = output()->data(outputDataID)->getDimensions();

    int outputCounter = 0;
    for (int i = 0; i < static_cast<int>(output()->vertices().size()); ++i) {
      if (output()->vertices()[i].isOwner()) {
        for (int dim = 0; dim < valueDim; ++dim) {
          output()->data(outputDataID)->values()[i * valueDim + dim] = receivedValues.at(outputCounter);
          ++outputCounter;
        }
      }
    }
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::mapConsistent(int inputDataID, int outputDataID)
{

  PRECICE_TRACE(inputDataID, outputDataID, polyparams);

  // Gather input data
  if (utils::MasterSlave::isSlave()) {
    // Input data is filtered
    auto localInDataFiltered = input()->getOwnedVertexData(inputDataID);
    int  localOutputSize     = output()->data(outputDataID)->values().size();

    // Send data and output size
    utils::MasterSlave::_communication->send(localInDataFiltered, 0);
    utils::MasterSlave::_communication->send(localOutputSize, 0);

  } else { // Master or Serial case

    int valueDim = output()->data(outputDataID)->getDimensions();

    std::vector<double> globalInValues((_matrixA.cols() - polyparams) * valueDim, 0.0);
    std::vector<int>    outValuesSize;

    if (utils::MasterSlave::isMaster()) { // Parallel case

      // Filter input data
      const auto &localInData = input()->getOwnedVertexData(inputDataID);
      std::copy(localInData.data(), localInData.data() + localInData.size(), globalInValues.begin());
      outValuesSize.push_back(output()->data(outputDataID)->values().size());

      int inputSizeCounter = localInData.size();
      int slaveOutDataSize{0};

      std::vector<double> slaveBuffer;

      for (Rank rank : utils::MasterSlave::allSlaves()) {
        utils::MasterSlave::_communication->receive(slaveBuffer, rank);
        std::copy(slaveBuffer.begin(), slaveBuffer.end(), globalInValues.begin() + inputSizeCounter);
        inputSizeCounter += slaveBuffer.size();

        utils::MasterSlave::_communication->receive(slaveOutDataSize, rank);
        outValuesSize.push_back(slaveOutDataSize);
      }

    } else { // Serial case
      const auto &localInData = input()->data(inputDataID)->values();
      std::copy(localInData.data(), localInData.data() + localInData.size(), globalInValues.begin());
      outValuesSize.push_back(output()->data(outputDataID)->values().size());
    }

    Eigen::VectorXd p(_matrixA.cols());   // rows == n
    Eigen::VectorXd in(_matrixA.cols());  // rows == n
    Eigen::VectorXd out(_matrixA.rows()); // rows == outputSize
    in.setZero();

    // Construct Eigen vectors
    Eigen::Map<Eigen::VectorXd> inputValues(globalInValues.data(), globalInValues.size());

    Eigen::VectorXd outputValues((_matrixA.rows()) * valueDim);
    outputValues.setZero();

    // For every data dimension, perform mapping
    for (int dim = 0; dim < valueDim; dim++) {
      // Fill input from input data values (last polyparams entries remain zero)
      for (int i = 0; i < in.size() - polyparams; i++) {
        in[i] = inputValues[i * valueDim + dim];
      }

      p   = _qrMatrixC.solve(in);
      out = _matrixA * p;

      // Copy mapped data to ouptut data values
      for (int i = 0; i < out.size(); i++) {
        outputValues[i * valueDim + dim] = out[i];
      }
    }

    output()->data(outputDataID)->values() = Eigen::Map<Eigen::VectorXd>(outputValues.data(), outValuesSize.at(0));

    // Data scattering to slaves
    int beginPoint = outValuesSize.at(0);

    if (utils::MasterSlave::isMaster()) {
      for (Rank rank : utils::MasterSlave::allSlaves()) {
        precice::span<const double> toSend{outputValues.data() + beginPoint, static_cast<size_t>(outValuesSize.at(rank))};
        utils::MasterSlave::_communication->send(toSend, rank);
        beginPoint += outValuesSize.at(rank);
      }
    }
  }
  if (utils::MasterSlave::isSlave()) {
    std::vector<double> receivedValues;
    utils::MasterSlave::_communication->receive(receivedValues, 0);
    output()->data(outputDataID)->values() = Eigen::Map<Eigen::VectorXd>(receivedValues.data(), receivedValues.size());
  }
  if (hasConstraint(SCALEDCONSISTENT)) {
    scaleConsistentMapping(inputDataID, outputDataID);
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshFirstRound()
{
  PRECICE_TRACE();
  mesh::PtrMesh filterMesh, otherMesh;
  if (hasConstraint(CONSERVATIVE)) {
    filterMesh = output(); // remote
    otherMesh  = input();  // local
  } else {
    filterMesh = input();  // remote
    otherMesh  = output(); // local
  }

  if (otherMesh->vertices().empty())
    return; // Ranks not at the interface should never hold interface vertices

  // Tags all vertices that are inside otherMesh's bounding box, enlarged by the support radius

  if (_basisFunction.hasCompactSupport()) {
    auto bb = otherMesh->getBoundingBox();
    bb.expandBy(_basisFunction.getSupportRadius());

    query::Index indexTree(filterMesh);
    auto         vertices = indexTree.getVerticesInsideBox(bb);
    std::for_each(vertices.begin(), vertices.end(), [&filterMesh](size_t v) { filterMesh->vertices()[v].tag(); });
  } else {
    filterMesh->tagAll();
  }
}

template <typename RADIAL_BASIS_FUNCTION_T>
void RadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::tagMeshSecondRound()
{
  PRECICE_TRACE();

  if (not _basisFunction.hasCompactSupport())
    return; // Tags should not be changed

  mesh::PtrMesh mesh; // The mesh we want to filter

  if (hasConstraint(CONSERVATIVE)) {
    mesh = output();
  } else {
    mesh = input();
  }

  mesh::BoundingBox bb(mesh->getDimensions());

  // Construct bounding box around all owned vertices
  for (mesh::Vertex &v : mesh->vertices()) {
    if (v.isOwner()) {
      PRECICE_ASSERT(v.isTagged()); // Should be tagged from the first round
      bb.expandBy(v);
    }
  }
  // Enlarge bb by support radius
  bb.expandBy(_basisFunction.getSupportRadius());
  query::Index indexTree(mesh);
  auto         vertices = indexTree.getVerticesInsideBox(bb);
  std::for_each(vertices.begin(), vertices.end(), [&mesh](size_t v) { mesh->vertices()[v].tag(); });
}

// ------- Non-Member Functions ---------

template <typename RADIAL_BASIS_FUNCTION_T>
static Eigen::MatrixXd buildMatrixCLU(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, std::vector<bool> deadAxis)
{
  int inputSize  = inputMesh.vertices().size();
  int dimensions = inputMesh.getDimensions();

  int deadDimensions = 0;
  for (int d = 0; d < dimensions; d++) {
    if (deadAxis[d])
      deadDimensions += 1;
  }

  int polyparams = 1 + dimensions - deadDimensions;
  PRECICE_ASSERT(inputSize >= 1 + polyparams, inputSize);
  int n = inputSize + polyparams; // Add linear polynom degrees

  Eigen::MatrixXd matrixCLU(n, n);
  matrixCLU.setZero();

  for (int i = 0; i < inputSize; ++i) {
    for (int j = i; j < inputSize; ++j) {
      const auto &u   = inputMesh.vertices()[i].getCoords();
      const auto &v   = inputMesh.vertices()[j].getCoords();
      matrixCLU(i, j) = basisFunction.evaluate(utils::reduceVector((u - v), deadAxis).norm());
    }

    const auto reduced = utils::reduceVector(inputMesh.vertices()[i].getCoords(), deadAxis);

    for (int dim = 0; dim < dimensions - deadDimensions; dim++) {
      matrixCLU(i, inputSize + 1 + dim) = reduced[dim];
    }
    matrixCLU(i, inputSize) = 1.0;
  }

  matrixCLU.triangularView<Eigen::Lower>() = matrixCLU.transpose();

  return matrixCLU;
}

template <typename RADIAL_BASIS_FUNCTION_T>
static Eigen::MatrixXd buildMatrixA(RADIAL_BASIS_FUNCTION_T basisFunction, const mesh::Mesh &inputMesh, const mesh::Mesh &outputMesh, std::vector<bool> deadAxis)
{
  int inputSize  = inputMesh.vertices().size();
  int outputSize = outputMesh.vertices().size();
  int dimensions = inputMesh.getDimensions();

  int deadDimensions = 0;
  for (int d = 0; d < dimensions; d++) {
    if (deadAxis[d])
      deadDimensions += 1;
  }

  int polyparams = 1 + dimensions - deadDimensions;
  PRECICE_ASSERT(inputSize >= 1 + polyparams, inputSize);
  int n = inputSize + polyparams; // Add linear polynom degrees

  Eigen::MatrixXd matrixA(outputSize, n);
  matrixA.setZero();

  // Fill _matrixA with values
  for (int i = 0; i < outputSize; ++i) {
    for (int j = 0; j < inputSize; ++j) {
      const auto &u = outputMesh.vertices()[i].getCoords();
      const auto &v = inputMesh.vertices()[j].getCoords();
      matrixA(i, j) = basisFunction.evaluate(utils::reduceVector((u - v), deadAxis).norm());
    }

    const auto reduced = utils::reduceVector(outputMesh.vertices()[i].getCoords(), deadAxis);

    for (int dim = 0; dim < dimensions - deadDimensions; dim++) {
      matrixA(i, inputSize + 1 + dim) = reduced[dim];
    }
    matrixA(i, inputSize) = 1.0;
  }
  return matrixA;
}

} // namespace mapping
} // namespace precice
