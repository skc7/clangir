//===- CIRModule.cpp - CIR dialect Python bindings -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "clang-c/CIR/Dialect/CIR.h"

extern "C" {
MlirDialectHandle mlirGetDialectHandle__cir__();
}

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

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

  //===--------------------------------------------------------------------===//
  // CIR Types
  //===--------------------------------------------------------------------===//

  auto intType = mlir_type_subclass(m, "IntType", mlirTypeIsACIRIntType);
  intType.def_classmethod(
      "get",
      [](const nb::object &cls, unsigned width, bool isSigned,
         MlirContext ctx) {
        return cls(mlirCIRIntTypeGet(ctx, width, isSigned));
      },
      "Create a CIR integer type", nb::arg("cls"), nb::arg("width"),
      nb::arg("is_signed") = true, nb::arg("context") = nb::none());

  auto boolType = mlir_type_subclass(m, "BoolType", mlirTypeIsACIRBoolType);
  boolType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRBoolTypeGet(ctx));
      },
      "Create a CIR bool type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto voidType = mlir_type_subclass(m, "VoidType", mlirTypeIsACIRVoidType);
  voidType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRVoidTypeGet(ctx));
      },
      "Create a CIR void type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto pointerType =
      mlir_type_subclass(m, "PointerType", mlirTypeIsACIRPointerType);
  pointerType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType pointee) {
        return cls(mlirCIRPointerTypeGet(pointee));
      },
      "Create a CIR pointer type", nb::arg("cls"), nb::arg("pointee"));

  auto arrayType = mlir_type_subclass(m, "ArrayType", mlirTypeIsACIRArrayType);
  arrayType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType, uint64_t size) {
        return cls(mlirCIRArrayTypeGet(elementType, size));
      },
      "Create a CIR array type", nb::arg("cls"), nb::arg("element_type"),
      nb::arg("size"));

  //===--------------------------------------------------------------------===//
  // CIR Float Types
  //===--------------------------------------------------------------------===//

  auto floatType = mlir_type_subclass(m, "FloatType", mlirTypeIsACIRFloatType);
  floatType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFloatTypeGet(ctx));
      },
      "Create a CIR single-precision float type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto doubleType =
      mlir_type_subclass(m, "DoubleType", mlirTypeIsACIRDoubleType);
  doubleType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRDoubleTypeGet(ctx));
      },
      "Create a CIR double-precision float type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto fp16Type = mlir_type_subclass(m, "FP16Type", mlirTypeIsACIRFP16Type);
  fp16Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFP16TypeGet(ctx));
      },
      "Create a CIR FP16 type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto bf16Type = mlir_type_subclass(m, "BF16Type", mlirTypeIsACIRBF16Type);
  bf16Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRBF16TypeGet(ctx));
      },
      "Create a CIR BFloat16 type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto fp80Type = mlir_type_subclass(m, "FP80Type", mlirTypeIsACIRFP80Type);
  fp80Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFP80TypeGet(ctx));
      },
      "Create a CIR FP80 type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  auto fp128Type = mlir_type_subclass(m, "FP128Type", mlirTypeIsACIRFP128Type);
  fp128Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFP128TypeGet(ctx));
      },
      "Create a CIR FP128 type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  //===--------------------------------------------------------------------===//
  // CIR Complex Type
  //===--------------------------------------------------------------------===//

  auto complexType =
      mlir_type_subclass(m, "ComplexType", mlirTypeIsACIRComplexType);
  complexType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType) {
        return cls(mlirCIRComplexTypeGet(elementType));
      },
      "Create a CIR complex type", nb::arg("cls"), nb::arg("element_type"));

  //===--------------------------------------------------------------------===//
  // CIR Function Type
  //===--------------------------------------------------------------------===//

  auto funcType = mlir_type_subclass(m, "FuncType", mlirTypeIsACIRFuncType);
  funcType.def_classmethod(
      "get",
      [](const nb::object &cls, nb::list inputs, MlirType returnType,
         bool isVarArg, MlirContext ctx) {
        std::vector<MlirType> inputTypes;
        inputTypes.reserve(nb::len(inputs));
        for (nb::handle input : inputs) {
          inputTypes.push_back(nb::cast<MlirType>(input));
        }
        return cls(mlirCIRFuncTypeGet(ctx, inputTypes.size(), inputTypes.data(),
                                      returnType, isVarArg));
      },
      "Create a CIR function type", nb::arg("cls"), nb::arg("inputs"),
      nb::arg("return_type"), nb::arg("is_vararg") = false,
      nb::arg("context") = nb::none());

  //===--------------------------------------------------------------------===//
  // CIR Vector Type
  //===--------------------------------------------------------------------===//

  auto vectorType =
      mlir_type_subclass(m, "VectorType", mlirTypeIsACIRVectorType);
  vectorType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType, uint64_t size) {
        return cls(mlirCIRVectorTypeGet(elementType, size));
      },
      "Create a CIR vector type", nb::arg("cls"), nb::arg("element_type"),
      nb::arg("size"));

  //===--------------------------------------------------------------------===//
  // CIR Record Type
  //===--------------------------------------------------------------------===//

  auto recordType =
      mlir_type_subclass(m, "RecordType", mlirTypeIsACIRRecordType);
  recordType.def_classmethod(
      "get",
      [](const nb::object &cls, nb::list members, bool packed, bool padded,
         bool kind, MlirContext ctx) {
        std::vector<MlirType> memberTypes;
        memberTypes.reserve(nb::len(members));
        for (nb::handle member : members) {
          memberTypes.push_back(nb::cast<MlirType>(member));
        }
        return cls(mlirCIRRecordTypeGet(
            ctx, memberTypes.size(), memberTypes.data(), packed, padded, kind));
      },
      "Create a CIR identified and complete record type", nb::arg("cls"),
      nb::arg("members"), nb::arg("packed") = false, nb::arg("padded") = false,
      nb::arg("kind") = false, nb::arg("context") = nb::none());

  //===--------------------------------------------------------------------===//
  // CIR Method Type
  //===--------------------------------------------------------------------===//

  auto methodType =
      mlir_type_subclass(m, "MethodType", mlirTypeIsACIRMethodType);
  methodType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType memberFuncTy, MlirType clsTy) {
        return cls(mlirCIRMethodTypeGet(memberFuncTy, clsTy));
      },
      "Create a CIR method type (pointer-to-member-function)", nb::arg("cls"),
      nb::arg("member_func_type"), nb::arg("class_type"));

  //===--------------------------------------------------------------------===//
  // CIR DataMember Type
  //===--------------------------------------------------------------------===//

  auto dataMemberType =
      mlir_type_subclass(m, "DataMemberType", mlirTypeIsACIRDataMemberType);
  dataMemberType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType memberTy, MlirType clsTy) {
        return cls(mlirCIRDataMemberTypeGet(memberTy, clsTy));
      },
      "Create a CIR data member type (pointer-to-data-member)", nb::arg("cls"),
      nb::arg("member_type"), nb::arg("class_type"));

  //===--------------------------------------------------------------------===//
  // CIR VPtr Type
  //===--------------------------------------------------------------------===//

  auto vptrType = mlir_type_subclass(m, "VPtrType", mlirTypeIsACIRVPtrType);
  vptrType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRVPtrTypeGet(ctx));
      },
      "Create a CIR vptr type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  //===--------------------------------------------------------------------===//
  // CIR Exception Type
  //===--------------------------------------------------------------------===//

  auto exceptionType =
      mlir_type_subclass(m, "ExceptionType", mlirTypeIsACIRExceptionType);
  exceptionType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRExceptionTypeGet(ctx));
      },
      "Create a CIR exception info type", nb::arg("cls"),
      nb::arg("context") = nb::none());

  //===--------------------------------------------------------------------===//
  // CIR Attributes
  //===--------------------------------------------------------------------===//

  auto intAttr =
      mlir_attribute_subclass(m, "IntAttr", mlirAttributeIsACIRIntAttr);
  intAttr.def_classmethod(
      "get",
      [](const nb::object &cls, int64_t value, MlirType type) {
        return cls(mlirCIRIntAttrGet(type, value));
      },
      "Create a CIR integer attribute", nb::arg("cls"), nb::arg("value"),
      nb::arg("type"));

  auto boolAttr =
      mlir_attribute_subclass(m, "BoolAttr", mlirAttributeIsACIRBoolAttr);
  boolAttr.def_classmethod(
      "get",
      [](const nb::object &cls, bool value, MlirType type) {
        return cls(mlirCIRBoolAttrGet(type, value));
      },
      "Create a CIR bool attribute", nb::arg("cls"), nb::arg("value"),
      nb::arg("type"));

  auto fpAttr = mlir_attribute_subclass(m, "FPAttr", mlirAttributeIsACIRFPAttr);
  fpAttr.def_classmethod(
      "get",
      [](const nb::object &cls, double value, MlirType type) {
        return cls(mlirCIRFPAttrGet(type, value));
      },
      "Create a CIR floating-point attribute", nb::arg("cls"), nb::arg("value"),
      nb::arg("type"));

  auto zeroAttr =
      mlir_attribute_subclass(m, "ZeroAttr", mlirAttributeIsACIRZeroAttr);
  zeroAttr.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType type) {
        return cls(mlirCIRZeroAttrGet(type));
      },
      "Create a CIR zero attribute", nb::arg("cls"), nb::arg("type"));
}
