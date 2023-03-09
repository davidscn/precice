#pragma once

#include <chrono>
#include <string>

namespace precice::profiling {

/// Tag to annotate fundamental events
struct FundamentalTag {
};

/// Convenience instance of the @ref FundamentalTag
static constexpr FundamentalTag Fundamental{};

/** Represents an event that can be started and stopped.
 *
 * Also allows to attach data in a key-value format using @ref addData()
 *
 * The event keeps minimal state. Events are passed to the @ref EventRegistry.
 */
class Event {
public:
  enum class State : bool {
    STOPPED = false,
    RUNNING = true
  };

  /// Default clock type. All other chrono types are derived from it.
  using Clock = std::chrono::steady_clock;

  /// Name used to identify the timer. Events of the same name are accumulated to
  std::string name;

  /// Creates a new event and starts it, unless autostart = false
  Event(std::string eventName, bool autostart = true)
      : Event(std::move(eventName), false, autostart) {}
  Event(std::string eventName, FundamentalTag, bool autostart = true)
      : Event(std::move(eventName), true, autostart) {}

  Event(Event &&) = default;
  Event &operator=(Event &&) = default;

  // Copies would lead to duplicate entries
  Event(const Event &other) = delete;
  Event &operator=(const Event &) = delete;

  /// Stops the event if it's running and report its times to the EventRegistry
  ~Event();

  /// Starts or restarts a stopped event.
  void start();

  /// Stops a running event.
  void stop();

  /// Adds named integer data, associated to an event.
  void addData(const std::string &key, int value);

private:
  Event(std::string eventName, bool fundamental, bool autostart);

  int   _eid;
  State _state = State::STOPPED;
  bool  _fundamental{false};
};

/// Class that changes the prefix in its scope
class ScopedEventPrefix {
public:
  ScopedEventPrefix(const std::string &name);

  ~ScopedEventPrefix();

  void pop();

private:
  std::string previousName = "";
};

} // namespace precice::profiling
