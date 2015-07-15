include(LibFindMacros)

# Dependencies
#libfind_package(gcop)
unset(GCOP_LIBRARY CACHE)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(GCOP_PKGCONF gcop)

# Include dir
find_path(GCOP_INCLUDE_DIR
  NAMES gcop/system.h
  PATHS ${GCOP_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
if( GCOP_FIND_COMPONENTS )# If no components specified manually specifiy all the components
else()
set(GCOP_FIND_COMPONENTS algos views systems est)
endif()

foreach(component ${GCOP_FIND_COMPONENTS})
	find_library(GCOP_LIB_${component}
	  NAMES gcop_${component}
	)
	set(GCOP_LIBRARY "${GCOP_LIBRARY};${GCOP_LIB_${component}}")
endforeach(component)
# Utils is common to all
find_library(GCOP_LIB_utils
  NAMES gcop_utils
)
set(GCOP_LIBRARY "${GCOP_LIBRARY};${GCOP_LIB_utils}")
# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(GCOP_PROCESS_INCLUDES GCOP_INCLUDE_DIR)
set(GCOP_PROCESS_LIBS GCOP_LIBRARY)
libfind_process(GCOP)

