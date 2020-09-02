//===-- OpenMP.cpp -- OpenACC directive lowering --------------------------===//
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

#include "flang/Lower/OpenACC.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

#define TODO() llvm_unreachable("not yet implemented")

static void genACC(Fortran::lower::AbstractConverter &absConv,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {

  const auto &beginLoopDirective =
      std::get<Fortran::parser::AccBeginLoopDirective>(loopConstruct.t);
  const auto &loopDirective =
      std::get<Fortran::parser::AccLoopDirective>(beginLoopDirective.t);

  if (loopDirective.v == llvm::acc::ACCD_loop) {
    auto &firOpBuilder = absConv.getFirOpBuilder();
    auto currentLocation = absConv.getCurrentLocation();
    llvm::ArrayRef<mlir::Type> argTy;
    mlir::ValueRange range;
    // Temporarly set to default 0 as operands are not generated yet.
    llvm::SmallVector<int32_t, 2> operandSegmentSizes(/*Size=*/7,
                                                      /*Value=*/0);
    auto loopOp =
        firOpBuilder.create<mlir::acc::LoopOp>(currentLocation, argTy, range);
    loopOp.setAttr(mlir::acc::LoopOp::getOperandSegmentSizeAttr(),
                   firOpBuilder.getI32VectorAttr(operandSegmentSizes));
    firOpBuilder.createBlock(&loopOp.getRegion());
    auto &block = loopOp.getRegion().back();
    firOpBuilder.setInsertionPointToStart(&block);
    // ensure the block is well-formed.
    firOpBuilder.create<mlir::acc::YieldOp>(currentLocation);

    // Add attribute extracted from clauses.
    const auto &accClauseList =
        std::get<Fortran::parser::AccClauseList>(beginLoopDirective.t);

    for (const auto &clause : accClauseList.v) {
      if (const auto *collapseClause =
              std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {

        const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
        const auto collapseValue = Fortran::evaluate::ToInt64(*expr);
        if (collapseValue.has_value()) {
          loopOp.setAttr(mlir::acc::LoopOp::getCollapseAttrName(),
                         firOpBuilder.getI64IntegerAttr(collapseValue.value()));
        }
      } else if (const auto *seqClause =
                     std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
        (void)seqClause;
      } else if (const auto *gangClause =
                     std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
        (void)gangClause;
      } else if (const auto *vectorClause =
                     std::get_if<Fortran::parser::AccClause::Vector>(
                         &clause.u)) {
        (void)vectorClause;
      } else if (const auto *workerClause =
                     std::get_if<Fortran::parser::AccClause::Worker>(
                         &clause.u)) {
        (void)workerClause;
      } else {
        TODO();
      }
    }

    // Place the insertion point to the start of the first block.
    firOpBuilder.setInsertionPointToStart(&block);
  }
}

void Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &acc) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCCombinedConstruct
                  &combinedConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {
            genACC(converter, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCRoutineConstruct
                  &routineConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            TODO();
          },
      },
      acc.u);
}
