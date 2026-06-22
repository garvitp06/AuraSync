#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GMP::libgmpxx" for configuration "Release"
set_property(TARGET GMP::libgmpxx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GMP::libgmpxx PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libgmpxx-4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libgmpxx-4.dll"
  )

list(APPEND _cmake_import_check_targets GMP::libgmpxx )
list(APPEND _cmake_import_check_files_for_GMP::libgmpxx "${_IMPORT_PREFIX}/lib/libgmpxx-4.lib" "${_IMPORT_PREFIX}/bin/libgmpxx-4.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
