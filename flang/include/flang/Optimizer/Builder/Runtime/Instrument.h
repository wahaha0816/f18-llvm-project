//===---- Instrument.h - generate instrumentation API calls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_INSTRUMENT_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_INSTRUMENT_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

void genSignalArrayCopy(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value copied);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H
