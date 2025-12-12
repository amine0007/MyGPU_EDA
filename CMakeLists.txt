cmake_minimum_required(VERSION 3.17)
project(MyGPU_EDA LANGUAGES CXX)

# --- 1. CONFIGURATION ANTLR (Parser) ---
# Trouver la librairie ANTLR installée via apt-get
find_library(ANTLR4_LIB antlr4-runtime)
find_path(ANTLR4_INCLUDE_DIR antlr4-runtime.h PATH_SUFFIXES antlr4-runtime)

# --- 2. CONFIGURATION SFML (Graphique) ---
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# --- 3. CONFIGURATION CUDA (Simulation) ---
# On vérifie si le compilateur CUDA (nvcc) est présent
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA trouvé ! Activation de la simulation GPU.")
    set(CUDA_SOURCES src/simulation/kernel.cu)
    add_definitions(-DUSE_CUDA) # On définit une macro pour le code C++
else()
    message(WARNING "CUDA non trouvé (Normal sur Codespaces). La simulation GPU sera désactivée.")
    set(CUDA_SOURCES "")
endif()

# --- 4. LISTE DES FICHIERS SOURCES ---
# (Note: Les fichiers générés par ANTLR seront ajoutés ici plus tard)
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
    ${ANTLR4_INCLUDE_DIR}
)

# --- 7. LINK DES LIBRAIRIES ---
target_link_libraries(MyEDA PRIVATE 
    sfml-graphics sfml-window sfml-system
    ${ANTLR4_LIB}
)

if(CUDAToolkit_FOUND)
    target_link_libraries(MyEDA PRIVATE CUDA::cudart)
endif()