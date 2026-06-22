#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GMP::libgmp" for configuration "Release"
set_property(TARGET GMP::libgmp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GMP::libgmp PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libgmp-10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libgmp-10.dll"
  )

list(APPEND _cmake_import_check_targets GMP::libgmp )
list(APPEND _cmake_import_check_files_for_GMP::libgmp "${_IMPORT_PREFIX}/lib/libgmp-10.lib" "${_IMPORT_PREFIX}/bin/libgmp-10.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
