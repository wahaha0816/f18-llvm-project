//===-- Optimizer/Support/InitFIR.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
#define FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

namespace fir::support {

/// Register and load all the dialects used by flang.
inline void registerAndLoadDialects(mlir::MLIRContext &ctx) {
  auto registry = ctx.getDialectRegistry();
  // clang-format off
  registry.insert<mlir::AffineDialect,
                  FIROpsDialect,
                  FIRCodeGenDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::acc::OpenACCDialect,
                  mlir::omp::OpenMPDialect,
                  mlir::scf::SCFDialect,
                  mlir::StandardOpsDialect,
                  mlir::vector::VectorDialect>();
  // clang-format on
  registry.loadAll(&ctx);
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerGeneralPasses() {
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerAffineLoopFusionPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLoopCoalescingPass();
  mlir::registerStripDebugInfoPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerInlinerPass();
  mlir::registerSCCPPass();
  mlir::registerMemRefDataFlowOptPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLocationSnapshotPass();
  mlir::registerAffinePipelineDataTransferPass();

  mlir::registerAffineVectorizePass();
  mlir::registerAffineLoopUnrollPass();
  mlir::registerAffineLoopUnrollAndJamPass();
  mlir::registerSimplifyAffineStructuresPass();
  mlir::registerAffineLoopInvariantCodeMotionPass();
  mlir::registerAffineLoopTilingPass();
  mlir::registerAffineDataCopyGenerationPass();

  mlir::registerConvertAffineToStandardPass();
}

inline void registerFIRPasses() { registerGeneralPasses(); }

} // namespace fir::support

#endif // FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
