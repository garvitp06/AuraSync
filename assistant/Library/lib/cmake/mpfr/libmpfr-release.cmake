#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MPFR::libmpfr" for configuration "Release"
set_property(TARGET MPFR::libmpfr APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MPFR::libmpfr PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libmpfr-6.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libmpfr-6.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS MPFR::libmpfr )
list(APPEND _IMPORT_CHECK_FILES_FOR_MPFR::libmpfr "${_IMPORT_PREFIX}/lib/libmpfr-6.lib" "${_IMPORT_PREFIX}/bin/libmpfr-6.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
