cmake_minimum_required(VERSION 3.24)
project(aistank LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- MuJoCo (prebuilt release: https://github.com/google-deepmind/mujoco/releases)
set(MUJOCO_DIR "" CACHE PATH "Path to extracted MuJoCo release (contains include/ and lib/)")
if(NOT MUJOCO_DIR)
    message(FATAL_ERROR "Set -DMUJOCO_DIR=C:/path/to/mujoco-3.x.x")
endif()

add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_DIR}/bin/mujoco.dll"
    IMPORTED_IMPLIB   "${MUJOCO_DIR}/lib/mujoco.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_DIR}/include")

# ---- Engine core DLL
add_library(aistank SHARED
    src/EngineCore.cpp
    src/SimulationLoop.cpp
    src/AistankApi.cpp)

target_include_directories(aistank PUBLIC include)
target_compile_definitions(aistank PRIVATE AISTANK_EXPORTS NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(aistank PRIVATE mujoco d3d12 dxgi)
target_compile_options(aistank PRIVATE /W4 /permissive- $<$<CONFIG:Release>:/O2 /fp:fast>)

# ---- HLSL compute shaders → .cso via DXC (one entry point per blob)
find_program(DXC_EXE dxc HINTS "$ENV{WindowsSdkVerBinPath}x64" "$ENV{VULKAN_SDK}/Bin")
if(NOT DXC_EXE)
    message(FATAL_ERROR "dxc.exe not found — install the Windows SDK or DXC release")
endif()

set(SHADER_SRC ${CMAKE_SOURCE_DIR}/shaders/AgentShaders.hlsl)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/$<CONFIG>/shaders)
set(SHADER_ENTRIES
    CS_GatherObservations
    CS_PolicyForward
    CS_RewardAndTerminate
    CS_ApplyResets
    CS_BuildInstances)

set(SHADER_BLOBS "")
foreach(ENTRY ${SHADER_ENTRIES})
    set(BLOB ${SHADER_OUT_DIR}/${ENTRY}.cso)
    add_custom_command(
        OUTPUT ${BLOB}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
        COMMAND ${DXC_EXE} -T cs_6_0 -E ${ENTRY} -O3
                $<$<CONFIG:Debug>:-Zi> $<$<CONFIG:Debug>:-Qembed_debug>
                -Fo ${BLOB} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "DXC ${ENTRY}")
    list(APPEND SHADER_BLOBS ${BLOB})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_BLOBS})
add_dependencies(aistank shaders)

# Stage mujoco.dll next to aistank.dll for the C# csproj copy step.
add_custom_command(TARGET aistank POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DIR}/bin/mujoco.dll" "$<TARGET_FILE_DIR:aistank>")
