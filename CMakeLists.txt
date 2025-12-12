cmake_minimum_required(VERSION 3.17)
project(MyGPU_EDA LANGUAGES CXX)

# --- 0. CORRECTIF WARNING CUDA ---
# Utilise l'architecture de TA carte graphique automatiquement
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# --- 1. CONFIGURATION ANTLR (DÉSACTIVÉ TEMPORAIREMENT POUR WINDOWS) ---
# find_library(ANTLR4_LIB antlr4-runtime)
# find_path(ANTLR4_INCLUDE_DIR antlr4-runtime.h PATH_SUFFIXES antlr4-runtime)

# --- 2. CONFIGURATION SFML (Graphique) ---
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# --- 3. CONFIGURATION CUDA (Simulation) ---
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA trouvé ! Activation de la simulation GPU.")
    set(CUDA_SOURCES src/simulation/kernel.cu)
    add_definitions(-DUSE_CUDA)
else()
    message(WARNING "CUDA non trouvé. La simulation GPU sera désactivée.")
    set(CUDA_SOURCES "")
endif()

# --- 4. FICHIERS SOURCES ---
set(SOURCES 
    src/main.cpp
    ${CUDA_SOURCES}
)

# --- 5. CRÉATION DE L'EXÉCUTABLE ---
add_executable(MyEDA ${SOURCES})

# --- 6. INCLUSION DES DOSSIERS ---
target_include_directories(MyEDA PRIVATE 
    include 
    src 
    # ${ANTLR4_INCLUDE_DIR}  <-- Désactivé
)

# --- 7. LINK DES LIBRAIRIES ---
target_link_libraries(MyEDA PRIVATE 
    sfml-graphics sfml-window sfml-system
    # ${ANTLR4_LIB}          <-- Désactivé
)

if(CUDAToolkit_FOUND)
    target_link_libraries(MyEDA PRIVATE CUDA::cudart)
endif()