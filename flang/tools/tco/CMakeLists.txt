set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
)

llvm_map_components_to_libnames(llvm_libs
  ${LLVM_LINK_COMPONENTS}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIROptimizer
  ${dialect_libs}
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToStandard
  MLIREDSC
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorToLLVM
  ${llvm_libs}
)

add_flang_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
target_link_libraries(tco PRIVATE ${LIBS})

