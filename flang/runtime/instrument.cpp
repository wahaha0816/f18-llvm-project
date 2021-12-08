//===-- runtime/instrument.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/instrument.h"
#include "flang/Runtime/descriptor.h"
#include <cstdio>

namespace Fortran::runtime {


extern "C" {
void RTNAME(SignalArrayCopy)(const Descriptor &copied,
    const char *sourceFile, int sourceLine) {
  std::fputs("array copy ", stderr);
  if (sourceFile) {
    std::fprintf(stderr, "at (%s", sourceFile);
    std::fprintf(stderr, ":%d)", sourceLine);
  }
  std::size_t copiedBytes{copied.Elements() * copied.ElementBytes()};
  std::fprintf(stderr, ": copied %zu bytes", copiedBytes);
  std::fputs("\n", stderr);
}

} // extern "C"
} // namespace Fortran::runtime
