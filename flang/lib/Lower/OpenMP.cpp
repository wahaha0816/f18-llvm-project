//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
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

#include "flang/Lower/OpenMP.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

#define TODO() llvm_unreachable("not yet implemented")

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &ABSConv,
    Fortran::lower::pft::Evaluation &Eval,
    const Fortran::parser::OpenMPConstruct &OMPConstruct) {
  if (auto StandaloneConstruct =
          std::get_if<Fortran::parser::OpenMPStandaloneConstruct>(
              &OMPConstruct.u)) {

    if (auto SimpleStandaloneConstruct =
            std::get_if<Fortran::parser::OpenMPSimpleStandaloneConstruct>(
                &StandaloneConstruct->u)) {
      const auto &Directive{
          std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
              SimpleStandaloneConstruct->t)};
      switch (Directive.v) {
      default:
        TODO();
      case parser::OmpSimpleStandaloneDirective::Directive::Barrier: {
        ABSConv.getFirOpBuilder().create<mlir::omp::BarrierOp>(
            ABSConv.getCurrentLocation());
        break;
      }
      }
    }
  }
}

void Fortran::lower::genOpenMPEndLoop(
    Fortran::lower::AbstractConverter &, Fortran::lower::pft::Evaluation &,
    const Fortran::parser::OmpEndLoopDirective &) {
  TODO();
}
