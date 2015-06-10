include(LibFindMacros)
unset(OBJECT_RENDERER_LIBS CACHE)

# Dependencies
#libfind_package(gcop)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(OBJECT_RENDERER_PKGCONF object_renderer)

# Include dir
find_path(OBJECT_RENDERER_INCLUDE_DIR
  NAMES object_renderer/ogre_application.h
  PATHS ${OBJECT_RENDERER_PKGCONF_INCLUDE_DIRS}
)

find_library(OBJECT_RENDERER_LIB_ogre_render
  NAMES ogre_render
  PATHS ${OBJECT_RENDERER_PKGCONF_LIBRARY_DIRS}
)
find_library(OBJECT_RENDERER_LIB_virtual_image_proc
  NAMES virtual_image_proc
  PATHS ${OBJECT_RENDERER_PKGCONF_LIBRARY_DIRS}
)
set(OBJECT_RENDERER_LIBS "${OBJECT_RENDERER_LIB_ogre_render};${OBJECT_RENDERER_LIB_virtual_image_proc}")

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(OBJECT_RENDERER_PROCESS_INCLUDES OBJECT_RENDERER_INCLUDE_DIR)
set(OBJECT_RENDERER_PROCESS_LIBS OBJECT_RENDERER_LIBS)
libfind_process(OBJECT_RENDERER)

