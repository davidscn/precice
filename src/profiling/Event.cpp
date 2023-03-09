#include "profiling/Event.hpp"
#include "profiling/EventUtils.hpp"
#include "utils/assertion.hpp"

namespace precice::profiling {

Event::Event(std::string eventName, bool fundamental, bool autostart)
    : _fundamental(fundamental)
{
  auto &er = EventRegistry::instance();
  _eid     = er.nameToID(EventRegistry::instance().prefix + eventName);
  if (autostart) {
    start();
  }
}

Event::~Event()
{
  if (_state == State::RUNNING) {
    stop();
  }
}

void Event::start()
{
  auto timestamp = Clock::now();
  PRECICE_ASSERT(_state == State::STOPPED, _eid);
  _state = State::RUNNING;

  if (EventRegistry::instance().accepting(toEventClass(_fundamental))) {
    EventRegistry::instance().put(StartEntry{_eid, timestamp});
  }
}

void Event::stop()
{
  auto timestamp = Clock::now();
  PRECICE_ASSERT(_state == State::RUNNING, _eid);
  _state = State::STOPPED;

  if (EventRegistry::instance().accepting(toEventClass(_fundamental))) {
    EventRegistry::instance().put(StopEntry{_eid, timestamp});
  }
}

void Event::addData(const std::string &key, int value)
{
  auto timestamp = Clock::now();
  PRECICE_ASSERT(_state == State::RUNNING, _eid);

  auto &er = EventRegistry::instance();
  if (er.accepting(toEventClass(_fundamental))) {
    auto did = er.nameToID(key);
    er.put(DataEntry{_eid, timestamp, did, value});
  }
}

// -----------------------------------------------------------------------

ScopedEventPrefix::ScopedEventPrefix(std::string const &name)
{
  previousName = EventRegistry::instance().prefix;
  EventRegistry::instance().prefix += name;
}

ScopedEventPrefix::~ScopedEventPrefix()
{
  pop();
}

void ScopedEventPrefix::pop()
{
  EventRegistry::instance().prefix = previousName;
}

} // namespace precice::profiling
