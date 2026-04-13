# Copyright 2024 The Manifold Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(GPU_COMMON_SRCS gpu_sort.cpp)

if(MANIFOLD_GPU_METAL AND APPLE)
  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)

  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    add_library(manifold_gpu
      gpu_hal_metal.mm
      gpu_hal_null.cpp  # compiled but gated by #ifndef
      ${GPU_COMMON_SRCS}
    )
    target_compile_definitions(manifold_gpu PUBLIC MANIFOLD_GPU_METAL)
    target_link_libraries(manifold_gpu PRIVATE
      ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    set_source_files_properties(gpu_hal_metal.mm PROPERTIES
      COMPILE_FLAGS "-fobjc-arc")

    # Copy Metal shader source for runtime compilation.
    set(METAL_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/sort.metal)
    add_custom_command(TARGET manifold_gpu POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${METAL_SHADER_SRC} ${CMAKE_BINARY_DIR}/sort.metal
      COMMENT "Copying Metal sort shader to build directory")

    # Try pre-compiling to metallib if Metal compiler is available.
    execute_process(
      COMMAND xcrun -sdk macosx metal --version
      RESULT_VARIABLE METAL_COMPILER_RESULT
      OUTPUT_QUIET ERROR_QUIET)
    if(METAL_COMPILER_RESULT EQUAL 0)
      set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/sort.air)
      set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/sort.metallib)
      add_custom_command(OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SRC} -o ${METAL_AIR}
        DEPENDS ${METAL_SHADER_SRC}
        COMMENT "Compiling Metal sort shader")
      add_custom_command(OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal sort shader library")
      add_custom_target(gpu_metal_shaders ALL DEPENDS ${METAL_LIB})
      add_dependencies(manifold_gpu gpu_metal_shaders)
      add_custom_command(TARGET manifold_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${METAL_LIB} ${CMAKE_BINARY_DIR}/sort.metallib
        COMMENT "Copying sort.metallib to build directory")
      message(STATUS "GPU Metal shaders: pre-compiled")
    else()
      message(STATUS "GPU Metal shaders: runtime compilation (install Metal Toolchain for pre-compilation)")
    endif()

    message(STATUS "GPU backend: Metal")
  else()
    message(STATUS "GPU backend: Metal framework not found, disabled")
    add_library(manifold_gpu gpu_hal_null.cpp ${GPU_COMMON_SRCS})
  endif()

elseif(MANIFOLD_GPU_OPENCL)
  find_package(OpenCL REQUIRED)
  add_library(manifold_gpu
    gpu_hal_opencl.cpp
    gpu_hal_null.cpp
    ${GPU_COMMON_SRCS}
  )
  target_compile_definitions(manifold_gpu PUBLIC MANIFOLD_GPU_OPENCL)
  target_link_libraries(manifold_gpu PRIVATE OpenCL::OpenCL)
  message(STATUS "GPU backend: OpenCL")

else()
  add_library(manifold_gpu gpu_hal_null.cpp ${GPU_COMMON_SRCS})
  message(STATUS "GPU backend: none (null stub)")
endif()

target_include_directories(manifold_gpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
target_compile_features(manifold_gpu PUBLIC cxx_std_17)

# Export alongside manifold so install(EXPORT manifoldTargets) works.
install(TARGETS manifold_gpu EXPORT manifoldTargets)

# Inherit compile options from parent manifold target so that
# MANIFOLD_PAR, MANIFOLD_DEBUG, etc. are defined consistently.
if(MANIFOLD_PAR)
  target_compile_options(manifold_gpu PRIVATE -DMANIFOLD_PAR=1)
else()
  target_compile_options(manifold_gpu PRIVATE -DMANIFOLD_PAR=-1)
endif()
foreach(OPT IN ITEMS MANIFOLD_DEBUG MANIFOLD_ASSERT MANIFOLD_CROSS_SECTION TRACY_ENABLE)
  if(${${OPT}})
    target_compile_options(manifold_gpu PRIVATE -D${OPT})
  endif()
endforeach()

# Link TBB if parallel backend is enabled (needed by parallel.h included via impl.h).
if(MANIFOLD_PAR)
  target_link_libraries(manifold_gpu PRIVATE $<TARGET_NAME_IF_EXISTS:TBB::tbb>)
endif()
