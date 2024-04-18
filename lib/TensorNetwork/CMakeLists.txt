add_mlir_dialect_library(MLIRTensorNetwork
    TensorNetworkTypes.cpp
    TensorNetworkDialect.cpp
    TensorNetworkOps.cpp
    TensorNetworkPasses.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TensorNetwork

    DEPENDS
    MLIRTensorNetworkOpsIncGen
    MLIRTensorNetworkPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR

)