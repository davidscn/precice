#pragma once

#include <Eigen/Core>

#include "DataContext.hpp"
#include "logging/Logger.hpp"

namespace precice {
namespace impl {

/**
 * @brief Stores one Data object with related mesh. Context stores data to be read from and potentially provides a read mapping.
 *
 * Derived from DataContext
 */
class ReadDataContext : public DataContext {
public:
  /**
   * @brief Construct a new ReadDataContext object without a mapping.
   * 
   * @param data Data associated with this ReadDataContext.
   * @param mesh Mesh associated with this ReadDataContext.
   */
  ReadDataContext(
      mesh::PtrData data,
      mesh::PtrMesh mesh);

  /**
   * @brief Links a MappingContext and the MeshContext required by the read mapping requires to this ReadDataContext.
   * 
   * A read mapping maps _fromData to _providedData. A ReadDataContext already has _providedData, but additionally requires _fromData.
   *
   * @param[in] mappingContext Context of read mapping
   * @param[in] meshContext Context of mesh this read mapping is mapping from (_fromData)
   */
  void configureMapping(const MappingContext &mappingContext, const MeshContext &meshContext) override;

private:
  static logging::Logger _log;
};

} // namespace impl
} // namespace precice