//===-- include/flang/Runtime/instrument.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef FORTRAN_RUNTIME_ASSIGN_H_
#define FORTRAN_RUNTIME_ASSIGN_H_

#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
class Descriptor;
extern "C" {
// API for lowering assignment
void RTNAME(SignalArrayCopy)(const Descriptor &copied,
    const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ASSIGN_H_
