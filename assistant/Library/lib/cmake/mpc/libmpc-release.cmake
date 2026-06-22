#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MPC::libmpc" for configuration "Release"
set_property(TARGET MPC::libmpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MPC::libmpc PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libmpc-3.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libmpc-3.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS MPC::libmpc )
list(APPEND _IMPORT_CHECK_FILES_FOR_MPC::libmpc "${_IMPORT_PREFIX}/lib/libmpc-3.lib" "${_IMPORT_PREFIX}/bin/libmpc-3.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
