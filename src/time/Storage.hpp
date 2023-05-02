#pragma once

#include <Eigen/Core>
#include "logging/Logger.hpp"

namespace precice::time {

class Storage {
public:
  /// Fixed time associated with beginning of window
  static const double WINDOW_START;

  /// Fixed time associated with end of window
  static const double WINDOW_END;

  /**
   * @brief Stores data samples in time and provides corresponding convenience functions.
   *
   * The Storage must be initialized before it can be used. Then values can be stored in the Storage. It is only allowed to store samples with increasing times. Overwriting existing samples or writing samples with a time smaller then the maximum stored time is forbidden.
   * The Storage is considered complete, when a sample with time 1.0 is provided. Then one can only sample or clear the storage, but not add any further samples.
   *
   * This Storage is used in the context of Waveform relaxation where samples in time are provided. Starting at the beginning of the window with time 0.0 and reaching the end of the window with time 1.0.
   */
  Storage();

  /**
   * @brief Initialize storage by storing given values at time 0.0 and 1.0.
   *
   * @param values initial values
   */
  void initialize(Eigen::VectorXd values);

  /**
   * @brief Store values at a specific time.
   *
   * It is only allowed to store values in time that come after values that were already stored. Therefore, time has to be larger than maxStoredNormalizedDt. Overwriting existing values is forbidden. The function clear() should be used to clear the storage and provide new values.
   *
   * @param time the time associated with the values
   * @param values stored values
   */
  void setValuesAtTime(double time, Eigen::VectorXd values);

  /**
   * @brief Get maximum normalized dt that is stored in this Storage.
   *
   * @return the maximum normalized dt from this Storage
   */
  double maxStoredNormalizedDt() const;

  /**
   * @brief Returns the values at time following "before" contained in this Storage.
   *
   * The stored normalized dt is larger or equal than "before". If "before" is a normalized dt stored in this Storage, this function returns the values at "before"
   *
   * @param before a double, where we want to find a normalized dt that comes directly after this one
   * @return Eigen::VectorXd values in this Storage at or directly after "before"
   */
  Eigen::VectorXd getValuesAtOrAfter(double before) const;

  /**
   * @brief Get all normalized dts stored in this Storage sorted ascending.
   *
   * @return Eigen::VectorXd containing all stored normalized dts in ascending order.
   */
  Eigen::VectorXd getTimes() const;

  /**
   * @brief Get all normalized dts and values in ascending order (with respect to normalized dts)
   *
   * @return std::pair<Eigen::VectorXd, Eigen::MatrixXd> containing all stored times and values in ascending order (with respect to normalized dts).
   */
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> getTimesAndValues() const;

  /**
   * @brief Number of stored times
   *
   * @return int number of stored times
   */
  int nTimes() const;

  /**
   * @brief Number of Dofs for each values
   *
   * @return int number of dofs
   */
  int nDofs() const;

  /**
   * @brief Move this Storage by storing the values at the end of the Storage at 0.0 and clearing the storage. Time 1.0 is initialized as values at 0.0
   */
  void move();

  /**
   * @brief Clear this Storage by deleting all values except values associated with 0.0.
   */
  void clear();

private:
  /// Stores values on the current window associated with normalized dt.
  std::vector<std::pair<double, Eigen::VectorXd>> _sampleStorage;

  mutable logging::Logger _log{"time::Storage"};
};

} // namespace precice::time
