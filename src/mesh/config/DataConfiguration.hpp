#pragma once

#include <string>
#include <vector>
#include "logging/Logger.hpp"
#include "mesh/Data.hpp"
#include "xml/XMLTag.hpp"

namespace precice {
namespace mesh {

/// Performs and provides configuration for Data objects from XML files.
class DataConfiguration : public xml::XMLTag::Listener {
public:
  struct ConfiguredData {
    std::string           name;
    int                   dimensions;
    Data::DataMappingType mappingType;

    ConfiguredData(
        const std::string &   name,
        int                   dimensions,
        Data::DataMappingType mappingType)
        : name(name), dimensions(dimensions), mappingType(mappingType) {}
  };

  DataConfiguration(xml::XMLTag &parent);

  void setDimensions(int dimensions);

  const std::vector<ConfiguredData> &data() const;

  ConfiguredData getRecentlyConfiguredData() const;

  virtual void xmlTagCallback(
      const xml::ConfigurationContext &context,
      xml::XMLTag &                    callingTag);

  virtual void xmlEndTagCallback(
      const xml::ConfigurationContext &context,
      xml::XMLTag &                    callingTag);

  /**
   * @brief Adds data manually.
   *
   * @param[in] name Unqiue name of the data.
   * @param[in] dataDimensions Dimensionality (1: scalar, 2,3: vector) of data.
   * @param[in] mappingType data mapping type (consistent or conservative)
   * Set a default here in order to compile the tests without any complains
   */
  void addData(const std::string &   name,
               int                   dataDimensions,
               Data::DataMappingType mappingType = Data::DataMappingType::NONE);

  //int getDimensions() const;

private:
  mutable logging::Logger _log{"mesh::DataConfiguration"};

  const std::string TAG               = "data";
  const std::string ATTR_NAME         = "name";
  const std::string ATTR_TYPE         = "type";
  const std::string TYPE_CONSISTENT   = "consistent";
  const std::string TYPE_CONSERVATIVE = "conservative";
  const std::string VALUE_VECTOR      = "vector";
  const std::string VALUE_SCALAR      = "scalar";

  /// Dimension of space.
  int _dimensions = 0;

  std::vector<ConfiguredData> _data;

  int _indexLastConfigured = -1;

  int getDataDimensions(const std::string &typeName) const;

  Data::DataMappingType getDataMappingType(const std::string &type) const;
};

} // namespace mesh
} // namespace precice
