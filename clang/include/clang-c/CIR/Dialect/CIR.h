/*===-- clang-c/CIR/Dialect/CIR.h - C API for CIR dialecte --------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides the C API for the CIR dialect.                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef CLANG_C_CIR_DIALECT_CIR_H
#define CLANG_C_CIR_DIALECT_CIR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CIR, cir);

#ifdef __cplusplus
}
#endif

#endif // CLANG_C_CIR_DIALECT_CIR_H
