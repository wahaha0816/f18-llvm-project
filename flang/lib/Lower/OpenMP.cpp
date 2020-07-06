//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenMP.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

#define TODO() llvm_unreachable("not yet implemented")

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &absConv,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &ompConstruct) {
  if (auto StandaloneConstruct{
          std::get_if<Fortran::parser::OpenMPStandaloneConstruct>(
              &ompConstruct.u)}) {

    if (auto SimpleStandaloneConstruct{
            std::get_if<Fortran::parser::OpenMPSimpleStandaloneConstruct>(
                &StandaloneConstruct->u)}) {
      const auto &Directive{
          std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
              SimpleStandaloneConstruct->t)};
      switch (Directive.v) {
      default:
        TODO();
      case parser::OmpSimpleStandaloneDirective::Directive::Barrier: {
        absConv.getFirOpBuilder().create<mlir::omp::BarrierOp>(
            absConv.getCurrentLocation());
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
