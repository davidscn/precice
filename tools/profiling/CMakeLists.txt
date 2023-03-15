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
set(_test_script "${CMAKE_CURRENT_LIST_DIR}/tests/simple-test.cmake")

foreach(test IN LISTS _tests)

  set(_test_name precice.events.merge.${test})
  set(_test_location "${CMAKE_CURRENT_LIST_DIR}/tests/${test}")
  set(_test_wd "${CMAKE_CURRENT_BINARY_DIR}/tests/${test}")
  file(MAKE_DIRECTORY ${_test_wd})
  message(STATUS "Test ${_test_name}")
  add_test(NAME ${_test_name}
    COMMAND ${CMAKE_COMMAND} -DEVENTS_SCRIPT=${_precice_events} -DFOLDER=${_test_location} -P ${_test_script}
    WORKING_DIRECTORY ${_test_wd}
    )
endforeach(test)

unset(_precice_events)
