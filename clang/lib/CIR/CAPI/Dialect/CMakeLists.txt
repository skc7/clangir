add_mlir_public_c_api_library(ClangCAPICIR
  CIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${CLANG_MAIN_INCLUDE_DIR}/clang-c/CIR

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRCIR
)

# Export the clang include directory so dependents can find clang-c headers
target_include_directories(ClangCAPICIR
  INTERFACE
  ${CLANG_MAIN_INCLUDE_DIR}
)

