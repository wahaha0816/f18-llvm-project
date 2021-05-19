
add_flang_library(FIRCodeGen
  CGOps.cpp
  CodeGen.cpp
  PreCGRewrite.cpp
  Target.cpp
  TargetRewrite.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  FIROptCodeGenPassIncGen
  CGOpsIncGen

  LINK_LIBS
  FIRDialect
  FIRSupport
  MLIROpenMPToLLVM
  MLIRTargetLLVMIRModuleTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
