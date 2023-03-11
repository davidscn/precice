#include "profiling/config/ProfilingConfiguration.hpp"
#include <boost/filesystem/path.hpp>
#include <cstdlib>
#include "logging/LogMacros.hpp"
#include "profiling/EventUtils.hpp"
#include "utils/assertion.hpp"
#include "xml/ConfigParser.hpp"
#include "xml/XMLAttribute.hpp"
#include "xml/XMLTag.hpp"

namespace precice::profiling {

ProfilingConfiguration::ProfilingConfiguration(xml::XMLTag &parent)
{
  using namespace xml;

  XMLTag tag(*this, "profiling", XMLTag::OCCUR_NOT_OR_ONCE);
  tag.setDocumentation("Allows to configure the profiling functionality of preCICE.");

  auto attrMode = makeXMLAttribute<std::string>("mode", "off")
                      .setOptions({"off", "fundamental", "all"})
                      .setDocumentation("Operaitonal modes of the profiling. "
                                        "\"fundamental\" will only write fundamental events. "
                                        "\"all\" writes all events.");
  tag.addAttribute(attrMode);

  auto attrFlush = makeXMLAttribute<int>("flush-every", 50)
                       .setDocumentation("Set the amount of events that should be kept in memory before flushing them to file. "
                                         "0 will only write at the end of the program. "
                                         "1 will write event directly to file. "
                                         "Everything larger than 1 will write events in blocks (recommended)");
  tag.addAttribute(attrFlush);

  auto attrDirectory = makeXMLAttribute<std::string>("directory", "..")
                           .setDocumentation("Directory to use as a root directory to  write the events to. "
                                             "Events will be written to `<directory>/precice-run/events/`");
  tag.addAttribute(attrDirectory);

  parent.addSubtag(tag);
}

void ProfilingConfiguration::xmlTagCallback(
    const xml::ConfigurationContext &context,
    xml::XMLTag &                    tag)
{
  auto mode       = tag.getStringAttributeValue("mode");
  auto flushEvery = tag.getIntAttributeValue("flush-every");
  auto directory  = boost::filesystem::path(tag.getStringAttributeValue("directory"));
  PRECICE_CHECK(flushEvery >= 0, "You configured the profiling to flush-every=\"{}\", which is invalid. "
                                 "Please choose a number >= 0.");

  using namespace precice;
  auto &er = profiling::EventRegistry::instance();

  er.setWriteQueueMax(flushEvery);
  directory /= "precice-run";
  directory /= "events";
  er.setDirectory(directory.string());

  if (mode == "off") {
    er.setMode(profiling::Mode::Off);
  } else if (mode == "fundamental") {
    er.setMode(profiling::Mode::Fundamental);
  } else if (mode == "all") {
    er.setMode(profiling::Mode::All);
  } else {
    PRECICE_UNREACHABLE("Unknown mode \"{}\"", mode);
  }
}

} // namespace precice::profiling
