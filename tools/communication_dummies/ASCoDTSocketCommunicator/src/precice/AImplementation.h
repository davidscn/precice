#ifndef PRECICE_AIMPLEMENTATION_H_
#define PRECICE_AIMPLEMENTATION_H_

//
// ASCoDT - Advanced Scientific Computing Development Toolkit
//
// This file was generated by ASCoDT's simplified SIDL compiler.
//
// Authors: Tobias Weinzierl, Atanas Atanasov
//

#include "precice/AAbstractImplementation.h"

#include <string>
#include <vector>

namespace precice {
class AImplementation;
}

class precice::AImplementation : public precice::AAbstractImplementation {
public:
  AImplementation();

  ~AImplementation();

  void
  main();

  void
  task();

  bool
  initialize();

  void
  acknowledge(int identifier, int& tag);

  void
  initialize(std::string const* addresses,
             int                addresses_size,
             int const*         vertexes,
             int                vertexes_size);

  bool
  send();

  bool
  send(double data, int index, int b_rank);

  void
  receive();

  void
  receive(double data, int index, int b_rank, int& tag);

  bool
  validate();

private:
  int const _rank;

  std::vector<std::string> _addresses;
  std::vector<int>         _vertexes;

  std::vector<std::string> _b_addresses;
  std::vector<int>         _b_vertexes;

  std::vector<double> _data;

  int _counter;

  bool _initialized;
  bool _received;

  // Prevents multiple overlapping executions of `task()`, what could occur by,
  // for example, rapidly clicking the start button for the current component
  // through ASCoDT GUI.
  pthread_mutex_t _task_mutex;

  pthread_mutex_t _initialize_mutex;
  pthread_cond_t  _initialize_cond;

  pthread_mutex_t _receive_mutex;
  pthread_cond_t  _receive_cond;
};

#endif
