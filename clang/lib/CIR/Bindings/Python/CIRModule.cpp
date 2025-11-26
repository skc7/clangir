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
  intType.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRIntType(type); },
      "Check if a type is a CIR integer type", nb::arg("type"));

  auto boolType = mlir_type_subclass(m, "BoolType", mlirTypeIsACIRBoolType);
  boolType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRBoolTypeGet(ctx));
      },
      "Create a CIR bool type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  boolType.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRBoolType(type); },
      "Check if a type is a CIR bool type", nb::arg("type"));

  auto voidType = mlir_type_subclass(m, "VoidType", mlirTypeIsACIRVoidType);
  voidType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRVoidTypeGet(ctx));
      },
      "Create a CIR void type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  voidType.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRVoidType(type); },
      "Check if a type is a CIR void type", nb::arg("type"));

  auto pointerType =
      mlir_type_subclass(m, "PointerType", mlirTypeIsACIRPointerType);
  pointerType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType pointee) {
        return cls(mlirCIRPointerTypeGet(pointee));
      },
      "Create a CIR pointer type", nb::arg("cls"), nb::arg("pointee"));
  pointerType.def_staticmethod(
      "isinstance",
      [](MlirType type) { return mlirTypeIsACIRPointerType(type); },
      "Check if a type is a CIR pointer type", nb::arg("type"));

  auto arrayType = mlir_type_subclass(m, "ArrayType", mlirTypeIsACIRArrayType);
  arrayType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType, uint64_t size) {
        return cls(mlirCIRArrayTypeGet(elementType, size));
      },
      "Create a CIR array type", nb::arg("cls"), nb::arg("element_type"),
      nb::arg("size"));
  arrayType.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRArrayType(type); },
      "Check if a type is a CIR array type", nb::arg("type"));

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
  floatType.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRFloatType(type); },
      "Check if a type is a CIR float type", nb::arg("type"));

  auto doubleType =
      mlir_type_subclass(m, "DoubleType", mlirTypeIsACIRDoubleType);
  doubleType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRDoubleTypeGet(ctx));
      },
      "Create a CIR double-precision float type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  doubleType.def_staticmethod(
      "isinstance",
      [](MlirType type) { return mlirTypeIsACIRDoubleType(type); },
      "Check if a type is a CIR double type", nb::arg("type"));

  auto fp16Type = mlir_type_subclass(m, "FP16Type", mlirTypeIsACIRFP16Type);
  fp16Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFP16TypeGet(ctx));
      },
      "Create a CIR FP16 type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  fp16Type.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRFP16Type(type); },
      "Check if a type is a CIR FP16 type", nb::arg("type"));

  auto bf16Type = mlir_type_subclass(m, "BF16Type", mlirTypeIsACIRBF16Type);
  bf16Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRBF16TypeGet(ctx));
      },
      "Create a CIR BFloat16 type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  bf16Type.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRBF16Type(type); },
      "Check if a type is a CIR BFloat16 type", nb::arg("type"));

  auto fp80Type = mlir_type_subclass(m, "FP80Type", mlirTypeIsACIRFP80Type);
  fp80Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFP80TypeGet(ctx));
      },
      "Create a CIR FP80 type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  fp80Type.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRFP80Type(type); },
      "Check if a type is a CIR FP80 type", nb::arg("type"));

  auto fp128Type = mlir_type_subclass(m, "FP128Type", mlirTypeIsACIRFP128Type);
  fp128Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirCIRFP128TypeGet(ctx));
      },
      "Create a CIR FP128 type", nb::arg("cls"),
      nb::arg("context") = nb::none());
  fp128Type.def_staticmethod(
      "isinstance", [](MlirType type) { return mlirTypeIsACIRFP128Type(type); },
      "Check if a type is a CIR FP128 type", nb::arg("type"));

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
  complexType.def_staticmethod(
      "isinstance",
      [](MlirType type) { return mlirTypeIsACIRComplexType(type); },
      "Check if a type is a CIR complex type", nb::arg("type"));

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
  funcType.def_property_readonly(
      "num_inputs",
      [](MlirType self) { return mlirCIRFuncTypeGetNumInputs(self); },
      "Get the number of input types");
  funcType.def_property_readonly(
      "inputs",
      [](MlirType self) {
        intptr_t numInputs = mlirCIRFuncTypeGetNumInputs(self);
        nb::list inputs;
        for (intptr_t i = 0; i < numInputs; ++i) {
          inputs.append(mlirCIRFuncTypeGetInput(self, i));
        }
        return inputs;
      },
      "Get the list of input types");
  funcType.def_property_readonly(
      "return_type",
      [](MlirType self) { return mlirCIRFuncTypeGetReturnType(self); },
      "Get the return type");
  funcType.def_property_readonly(
      "is_vararg", [](MlirType self) { return mlirCIRFuncTypeIsVarArg(self); },
      "Check if the function type is variadic");

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

  //===--------------------------------------------------------------------===//
  // CIR VisibilityAttr
  //===--------------------------------------------------------------------===//

  nb::enum_<MlirCIRVisibilityKind>(m, "VisibilityKind")
      .value("Default", MlirCIRVisibilityKindDefault)
      .value("Hidden", MlirCIRVisibilityKindHidden)
      .value("Protected", MlirCIRVisibilityKindProtected)
      .export_values();

  auto visibilityAttr = mlir_attribute_subclass(
      m, "VisibilityAttr", mlirAttributeIsACIRVisibilityAttr);
  visibilityAttr.def_classmethod(
      "get",
      [](const nb::object &cls, MlirCIRVisibilityKind kind, MlirContext ctx) {
        return cls(mlirCIRVisibilityAttrGet(ctx, kind));
      },
      "Create a CIR visibility attribute", nb::arg("cls"), nb::arg("kind"),
      nb::arg("context") = nb::none());
  visibilityAttr.def_property_readonly(
      "kind",
      [](MlirAttribute self) { return mlirCIRVisibilityAttrGetKind(self); },
      "Get the visibility kind");

  //===--------------------------------------------------------------------===//
  // CIR ExtraFuncAttributesAttr
  //===--------------------------------------------------------------------===//

  auto extraFuncAttributesAttr = mlir_attribute_subclass(
      m, "ExtraFuncAttributesAttr", mlirAttributeIsACIRExtraFuncAttributesAttr);
  extraFuncAttributesAttr.def_classmethod(
      "get",
      [](const nb::object &cls, MlirAttribute dictAttr) {
        return cls(mlirCIRExtraFuncAttributesAttrGet(dictAttr));
      },
      "Create a CIR extra function attributes attribute", nb::arg("cls"),
      nb::arg("dict_attr"));
  extraFuncAttributesAttr.def_property_readonly(
      "elements",
      [](MlirAttribute self) {
        return mlirCIRExtraFuncAttributesAttrGetElements(self);
      },
      "Get the dictionary of extra attributes");

  //===--------------------------------------------------------------------===//
  // CIR GlobalLinkageKind
  //===--------------------------------------------------------------------===//

  nb::enum_<MlirCIRGlobalLinkageKind>(m, "GlobalLinkageKind")
      .value("ExternalLinkage", MlirCIRGlobalLinkageKindExternalLinkage)
      .value("AvailableExternallyLinkage",
             MlirCIRGlobalLinkageKindAvailableExternallyLinkage)
      .value("LinkOnceAnyLinkage", MlirCIRGlobalLinkageKindLinkOnceAnyLinkage)
      .value("LinkOnceODRLinkage", MlirCIRGlobalLinkageKindLinkOnceODRLinkage)
      .value("WeakAnyLinkage", MlirCIRGlobalLinkageKindWeakAnyLinkage)
      .value("WeakODRLinkage", MlirCIRGlobalLinkageKindWeakODRLinkage)
      .value("InternalLinkage", MlirCIRGlobalLinkageKindInternalLinkage)
      .value("PrivateLinkage", MlirCIRGlobalLinkageKindPrivateLinkage)
      .value("ExternalWeakLinkage", MlirCIRGlobalLinkageKindExternalWeakLinkage)
      .value("CommonLinkage", MlirCIRGlobalLinkageKindCommonLinkage)
      .export_values();

  auto globalLinkageKindAttr = mlir_attribute_subclass(
      m, "GlobalLinkageKindAttr", mlirAttributeIsACIRGlobalLinkageKindAttr);
  globalLinkageKindAttr.def_classmethod(
      "get",
      [](const nb::object &cls, MlirCIRGlobalLinkageKind kind,
         MlirContext ctx) {
        return cls(mlirCIRGlobalLinkageKindAttrGet(ctx, kind));
      },
      "Create a CIR global linkage kind attribute", nb::arg("cls"),
      nb::arg("kind"), nb::arg("context") = nb::none());
  globalLinkageKindAttr.def_property_readonly(
      "kind",
      [](MlirAttribute self) {
        return mlirCIRGlobalLinkageKindAttrGetKind(self);
      },
      "Get the global linkage kind");

  //===--------------------------------------------------------------------===//
  // CIR CallingConv
  //===--------------------------------------------------------------------===//

  nb::enum_<MlirCIRCallingConv>(m, "CallingConv")
      .value("C", MlirCIRCallingConvC)
      .value("SpirKernel", MlirCIRCallingConvSpirKernel)
      .value("SpirFunction", MlirCIRCallingConvSpirFunction)
      .value("OpenCLKernel", MlirCIRCallingConvOpenCLKernel)
      .value("PTXKernel", MlirCIRCallingConvPTXKernel)
      .export_values();

  auto callingConvAttr = mlir_attribute_subclass(
      m, "CallingConvAttr", mlirAttributeIsACIRCallingConvAttr);
  callingConvAttr.def_classmethod(
      "get",
      [](const nb::object &cls, MlirCIRCallingConv conv, MlirContext ctx) {
        return cls(mlirCIRCallingConvAttrGet(ctx, conv));
      },
      "Create a CIR calling convention attribute", nb::arg("cls"),
      nb::arg("conv"), nb::arg("context") = nb::none());
  callingConvAttr.def_property_readonly(
      "conv",
      [](MlirAttribute self) { return mlirCIRCallingConvAttrGetConv(self); },
      "Get the calling convention");
}
