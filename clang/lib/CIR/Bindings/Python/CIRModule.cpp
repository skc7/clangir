//===- CIRModule.cpp - CIR dialect Python bindings -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

extern "C" {
MlirDialectHandle mlirGetDialectHandle__cir__();
}

namespace nb = nanobind;

NB_MODULE(_cirDialect, m) {
  m.doc() = "CIR dialect Python bindings";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__cir__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true,
      "Register the CIR dialect with a context");
}
