#
# Script to test event-precice merge trace and export given a folder
#
# Inputs:
#   FOLDER - the folder argument passed to merge
#   EVENTS_SCRIPT - the path to the precice-events script

find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(NOT FOLDER OR NOT EVENTS_SCRIPT)
  message(FATAL "Missing arguments")
endif()

message(STATUS "Removing previous files")
file(REMOVE events.json trace.json exports.csv)

message(STATUS "Testing: merge")
execute_process(COMMAND ${Python3_EXECUTABLE} ${EVENTS_SCRIPT} merge ${FOLDER}
  COMMAND_ECHO STDOUT)
if(NOT EXISTS "events.json")
  message(FATAL "No events.json file found")
endif()

message(STATUS "Testing: trace")
execute_process(COMMAND ${Python3_EXECUTABLE} ${EVENTS_SCRIPT} trace
  COMMAND_ECHO STDOUT
  )
if(NOT EXISTS "trace.json")
  message(FATAL "No trace.json file found")
endif()

message(STATUS "Testing: export")
execute_process(COMMAND ${Python3_EXECUTABLE} ${EVENTS_SCRIPT} export
  COMMAND_ECHO STDOUT)
if(NOT EXISTS "events.csv")
  message(FATAL "No events.csv file found")
endif()
