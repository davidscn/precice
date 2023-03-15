# Adds the tests for the precice-events

set(_tests
  from-exchange-dir
  from-precice-run
  from-solver-dir
  one-parallel-solver
  one-serial-damaged
  one-serial-solver
  one-solver-multiple-runs
  two-mixed-solvers
  two-parallel-solvers
  two-serial-solvers
  )

set(_precice_events "${CMAKE_CURRENT_LIST_DIR}/precice-events")

foreach(test IN LISTS _tests)

  set(_test_name precice.events.merge.${test})
  set(_test_location "${CMAKE_CURRENT_LIST_DIR}/tests/${test}")
  set(_test_wd "${CMAKE_CURRENT_BINARY_DIR}/tests/${test}")
  file(MAKE_DIRECTORY ${_test_wd})
  # set(_test_out "${CMAKE_CURRENT_LIST_DIR}/tests/${test}/events.json")
  message(STATUS "Test ${_test_name}")
  add_test(NAME ${_test_name}
    COMMAND ${_precice_events} merge ${_test_location}
    WORKING_DIRECTORY ${_test_wd}
    )
endforeach(test)

unset(_precice_events)
