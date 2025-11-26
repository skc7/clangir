//===- CIR.cpp - C Interface for CIR dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-c/CIR/Dialect/CIR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CIR, cir, cir::CIRDialect)

//===----------------------------------------------------------------------===//
// CIR Type API
//===----------------------------------------------------------------------===//

MlirType mlirCIRIntTypeGet(MlirContext ctx, unsigned width, bool isSigned) {
  return wrap(cir::IntType::get(unwrap(ctx), width, isSigned));
}

bool mlirTypeIsACIRIntType(MlirType type) {
  return llvm::isa<cir::IntType>(unwrap(type));
}

MlirType mlirCIRBoolTypeGet(MlirContext ctx) {
  return wrap(cir::BoolType::get(unwrap(ctx)));
}

bool mlirTypeIsACIRBoolType(MlirType type) {
  return llvm::isa<cir::BoolType>(unwrap(type));
}

MlirType mlirCIRVoidTypeGet(MlirContext ctx) {
  return wrap(cir::VoidType::get(unwrap(ctx)));
}

bool mlirTypeIsACIRVoidType(MlirType type) {
  return llvm::isa<cir::VoidType>(unwrap(type));
}

MlirType mlirCIRPointerTypeGet(MlirType pointee) {
  return wrap(cir::PointerType::get(unwrap(pointee)));
}

bool mlirTypeIsACIRPointerType(MlirType type) {
  return llvm::isa<cir::PointerType>(unwrap(type));
}

MlirType mlirCIRArrayTypeGet(MlirType elementType, uint64_t size) {
  return wrap(cir::ArrayType::get(unwrap(elementType).getContext(),
                                  unwrap(elementType), size));
}

bool mlirTypeIsACIRArrayType(MlirType type) {
  return llvm::isa<cir::ArrayType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Float Types
//===----------------------------------------------------------------------===//

MlirType mlirCIRFloatTypeGet(MlirContext ctx) {
  return wrap(cir::SingleType::get(unwrap(ctx)));
}

MlirType mlirCIRDoubleTypeGet(MlirContext ctx) {
  return wrap(cir::DoubleType::get(unwrap(ctx)));
}

MlirType mlirCIRFP16TypeGet(MlirContext ctx) {
  return wrap(cir::FP16Type::get(unwrap(ctx)));
}

MlirType mlirCIRBF16TypeGet(MlirContext ctx) {
  return wrap(cir::BF16Type::get(unwrap(ctx)));
}

MlirType mlirCIRFP80TypeGet(MlirContext ctx) {
  return wrap(cir::FP80Type::get(unwrap(ctx)));
}

MlirType mlirCIRFP128TypeGet(MlirContext ctx) {
  return wrap(cir::FP128Type::get(unwrap(ctx)));
}

bool mlirTypeIsACIRFloatType(MlirType type) {
  return llvm::isa<cir::SingleType>(unwrap(type));
}

bool mlirTypeIsACIRDoubleType(MlirType type) {
  return llvm::isa<cir::DoubleType>(unwrap(type));
}

bool mlirTypeIsACIRFP16Type(MlirType type) {
  return llvm::isa<cir::FP16Type>(unwrap(type));
}

bool mlirTypeIsACIRBF16Type(MlirType type) {
  return llvm::isa<cir::BF16Type>(unwrap(type));
}

bool mlirTypeIsACIRFP80Type(MlirType type) {
  return llvm::isa<cir::FP80Type>(unwrap(type));
}

bool mlirTypeIsACIRFP128Type(MlirType type) {
  return llvm::isa<cir::FP128Type>(unwrap(type));
}

bool mlirTypeIsACIRLongDoubleType(MlirType type) {
  return llvm::isa<cir::LongDoubleType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Complex Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRComplexTypeGet(MlirType elementType) {
  return wrap(cir::ComplexType::get(unwrap(elementType)));
}

bool mlirTypeIsACIRComplexType(MlirType type) {
  return llvm::isa<cir::ComplexType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Function Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRFuncTypeGet(MlirContext ctx, intptr_t numInputs,
                            MlirType const *inputs, MlirType returnType,
                            bool isVarArg) {
  SmallVector<mlir::Type, 4> inputTypes;
  ArrayRef<MlirType> inputsRef(inputs, numInputs);
  inputTypes.reserve(numInputs);
  for (MlirType input : inputsRef)
    inputTypes.push_back(unwrap(input));

  return wrap(cir::FuncType::get(inputTypes, unwrap(returnType), isVarArg));
}

bool mlirTypeIsACIRFuncType(MlirType type) {
  return llvm::isa<cir::FuncType>(unwrap(type));
}

intptr_t mlirCIRFuncTypeGetNumInputs(MlirType type) {
  auto funcType = mlir::cast<cir::FuncType>(unwrap(type));
  return static_cast<intptr_t>(funcType.getNumInputs());
}

MlirType mlirCIRFuncTypeGetInput(MlirType type, intptr_t pos) {
  auto funcType = mlir::cast<cir::FuncType>(unwrap(type));
  return wrap(funcType.getInput(static_cast<unsigned>(pos)));
}

MlirType mlirCIRFuncTypeGetReturnType(MlirType type) {
  auto funcType = mlir::cast<cir::FuncType>(unwrap(type));
  return wrap(funcType.getReturnType());
}

bool mlirCIRFuncTypeIsVarArg(MlirType type) {
  auto funcType = mlir::cast<cir::FuncType>(unwrap(type));
  return funcType.isVarArg();
}

//===----------------------------------------------------------------------===//
// CIR Vector Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRVectorTypeGet(MlirType elementType, uint64_t size) {
  return wrap(cir::VectorType::get(unwrap(elementType), size));
}

bool mlirTypeIsACIRVectorType(MlirType type) {
  return llvm::isa<cir::VectorType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Record Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRRecordTypeGet(MlirContext ctx, intptr_t numMembers,
                              MlirType const *members, bool packed, bool padded,
                              bool kind) {
  mlir::MLIRContext *context = unwrap(ctx);
  SmallVector<mlir::Type, 4> memberTypes;
  ArrayRef<MlirType> membersRef(members, numMembers);
  memberTypes.reserve(numMembers);
  for (MlirType member : membersRef)
    memberTypes.push_back(unwrap(member));

  cir::RecordType::RecordKind recordKind =
      kind ? cir::RecordType::RecordKind::Class
           : cir::RecordType::RecordKind::Struct;

  // Use the internal $_get method for identified complete records
  return wrap(
      cir::RecordType::get(context, memberTypes, packed, padded, recordKind));
}

bool mlirTypeIsACIRRecordType(MlirType type) {
  return llvm::isa<cir::RecordType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Method Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRMethodTypeGet(MlirType memberFuncTy, MlirType clsTy) {
  auto funcType = mlir::cast<cir::FuncType>(unwrap(memberFuncTy));
  auto recordType = mlir::cast<cir::RecordType>(unwrap(clsTy));
  return wrap(cir::MethodType::get(funcType, recordType));
}

bool mlirTypeIsACIRMethodType(MlirType type) {
  return llvm::isa<cir::MethodType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR DataMember Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRDataMemberTypeGet(MlirType memberTy, MlirType clsTy) {
  auto recordType = mlir::cast<cir::RecordType>(unwrap(clsTy));
  return wrap(cir::DataMemberType::get(unwrap(memberTy), recordType));
}

bool mlirTypeIsACIRDataMemberType(MlirType type) {
  return llvm::isa<cir::DataMemberType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR VPtr Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRVPtrTypeGet(MlirContext ctx) {
  return wrap(cir::VPtrType::get(unwrap(ctx)));
}

bool mlirTypeIsACIRVPtrType(MlirType type) {
  return llvm::isa<cir::VPtrType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Exception Type
//===----------------------------------------------------------------------===//

MlirType mlirCIRExceptionTypeGet(MlirContext ctx) {
  return wrap(cir::ExceptionInfoType::get(unwrap(ctx)));
}

bool mlirTypeIsACIRExceptionType(MlirType type) {
  return llvm::isa<cir::ExceptionInfoType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// CIR Attribute API
//===----------------------------------------------------------------------===//

MlirAttribute mlirCIRIntAttrGet(MlirType type, int64_t value) {
  // IntAttr has a builder that takes (type, int64_t) and infers width from type
  return wrap(cir::IntAttr::get(unwrap(type), value));
}

bool mlirAttributeIsACIRIntAttr(MlirAttribute attr) {
  return llvm::isa<cir::IntAttr>(unwrap(attr));
}

MlirAttribute mlirCIRBoolAttrGet(MlirType type, bool value) {
  // BoolAttr has a builder that takes (context, bool) but we accept type for
  // API consistency Extract context from type and use it
  return wrap(cir::BoolAttr::get(unwrap(type).getContext(), value));
}

bool mlirAttributeIsACIRBoolAttr(MlirAttribute attr) {
  return llvm::isa<cir::BoolAttr>(unwrap(attr));
}

MlirAttribute mlirCIRFPAttrGet(MlirType type, double value) {
  mlir::Type mlirType = unwrap(type);
  // Get the float semantics from the type to ensure they match
  auto fpType = mlir::cast<cir::FPTypeInterface>(mlirType);
  const llvm::fltSemantics &targetSemantics = fpType.getFloatSemantics();

  // Create APFloat from double (uses IEEEdouble semantics)
  APFloat apValue(value);

  // Convert to target semantics if needed
  if (&targetSemantics != &apValue.getSemantics()) {
    bool losesInfo;
    apValue.convert(targetSemantics, APFloat::rmNearestTiesToEven, &losesInfo);
  }

  return wrap(cir::FPAttr::get(mlirType, apValue));
}

bool mlirAttributeIsACIRFPAttr(MlirAttribute attr) {
  return llvm::isa<cir::FPAttr>(unwrap(attr));
}

MlirAttribute mlirCIRZeroAttrGet(MlirType type) {
  return wrap(cir::ZeroAttr::get(unwrap(type)));
}

bool mlirAttributeIsACIRZeroAttr(MlirAttribute attr) {
  return llvm::isa<cir::ZeroAttr>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// CIR VisibilityAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCIRVisibilityAttrGet(MlirContext ctx,
                                       MlirCIRVisibilityKind kind) {
  cir::VisibilityKind cirKind;
  switch (kind) {
  case MlirCIRVisibilityKindDefault:
    cirKind = cir::VisibilityKind::Default;
    break;
  case MlirCIRVisibilityKindHidden:
    cirKind = cir::VisibilityKind::Hidden;
    break;
  case MlirCIRVisibilityKindProtected:
    cirKind = cir::VisibilityKind::Protected;
    break;
  }
  return wrap(cir::VisibilityAttr::get(unwrap(ctx), cirKind));
}

bool mlirAttributeIsACIRVisibilityAttr(MlirAttribute attr) {
  return llvm::isa<cir::VisibilityAttr>(unwrap(attr));
}

MlirCIRVisibilityKind mlirCIRVisibilityAttrGetKind(MlirAttribute attr) {
  auto visAttr = mlir::cast<cir::VisibilityAttr>(unwrap(attr));
  switch (visAttr.getValue()) {
  case cir::VisibilityKind::Default:
    return MlirCIRVisibilityKindDefault;
  case cir::VisibilityKind::Hidden:
    return MlirCIRVisibilityKindHidden;
  case cir::VisibilityKind::Protected:
    return MlirCIRVisibilityKindProtected;
  }
  llvm_unreachable("Unknown VisibilityKind");
}

//===----------------------------------------------------------------------===//
// CIR ExtraFuncAttributesAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCIRExtraFuncAttributesAttrGet(MlirAttribute dictAttr) {
  auto dict = mlir::cast<mlir::DictionaryAttr>(unwrap(dictAttr));
  return wrap(cir::ExtraFuncAttributesAttr::get(dict));
}

bool mlirAttributeIsACIRExtraFuncAttributesAttr(MlirAttribute attr) {
  return llvm::isa<cir::ExtraFuncAttributesAttr>(unwrap(attr));
}

MlirAttribute mlirCIRExtraFuncAttributesAttrGetElements(MlirAttribute attr) {
  auto extraAttr = mlir::cast<cir::ExtraFuncAttributesAttr>(unwrap(attr));
  return wrap(extraAttr.getElements());
}

//===----------------------------------------------------------------------===//
// CIR GlobalLinkageKind
//===----------------------------------------------------------------------===//

MlirAttribute mlirCIRGlobalLinkageKindAttrGet(MlirContext ctx,
                                              MlirCIRGlobalLinkageKind kind) {
  cir::GlobalLinkageKind cirKind;
  switch (kind) {
  case MlirCIRGlobalLinkageKindExternalLinkage:
    cirKind = cir::GlobalLinkageKind::ExternalLinkage;
    break;
  case MlirCIRGlobalLinkageKindAvailableExternallyLinkage:
    cirKind = cir::GlobalLinkageKind::AvailableExternallyLinkage;
    break;
  case MlirCIRGlobalLinkageKindLinkOnceAnyLinkage:
    cirKind = cir::GlobalLinkageKind::LinkOnceAnyLinkage;
    break;
  case MlirCIRGlobalLinkageKindLinkOnceODRLinkage:
    cirKind = cir::GlobalLinkageKind::LinkOnceODRLinkage;
    break;
  case MlirCIRGlobalLinkageKindWeakAnyLinkage:
    cirKind = cir::GlobalLinkageKind::WeakAnyLinkage;
    break;
  case MlirCIRGlobalLinkageKindWeakODRLinkage:
    cirKind = cir::GlobalLinkageKind::WeakODRLinkage;
    break;
  case MlirCIRGlobalLinkageKindInternalLinkage:
    cirKind = cir::GlobalLinkageKind::InternalLinkage;
    break;
  case MlirCIRGlobalLinkageKindPrivateLinkage:
    cirKind = cir::GlobalLinkageKind::PrivateLinkage;
    break;
  case MlirCIRGlobalLinkageKindExternalWeakLinkage:
    cirKind = cir::GlobalLinkageKind::ExternalWeakLinkage;
    break;
  case MlirCIRGlobalLinkageKindCommonLinkage:
    cirKind = cir::GlobalLinkageKind::CommonLinkage;
    break;
  }
  return wrap(cir::GlobalLinkageKindAttr::get(unwrap(ctx), cirKind));
}

bool mlirAttributeIsACIRGlobalLinkageKindAttr(MlirAttribute attr) {
  return llvm::isa<cir::GlobalLinkageKindAttr>(unwrap(attr));
}

MlirCIRGlobalLinkageKind
mlirCIRGlobalLinkageKindAttrGetKind(MlirAttribute attr) {
  auto linkageAttr = mlir::cast<cir::GlobalLinkageKindAttr>(unwrap(attr));
  switch (linkageAttr.getValue()) {
  case cir::GlobalLinkageKind::ExternalLinkage:
    return MlirCIRGlobalLinkageKindExternalLinkage;
  case cir::GlobalLinkageKind::AvailableExternallyLinkage:
    return MlirCIRGlobalLinkageKindAvailableExternallyLinkage;
  case cir::GlobalLinkageKind::LinkOnceAnyLinkage:
    return MlirCIRGlobalLinkageKindLinkOnceAnyLinkage;
  case cir::GlobalLinkageKind::LinkOnceODRLinkage:
    return MlirCIRGlobalLinkageKindLinkOnceODRLinkage;
  case cir::GlobalLinkageKind::WeakAnyLinkage:
    return MlirCIRGlobalLinkageKindWeakAnyLinkage;
  case cir::GlobalLinkageKind::WeakODRLinkage:
    return MlirCIRGlobalLinkageKindWeakODRLinkage;
  case cir::GlobalLinkageKind::InternalLinkage:
    return MlirCIRGlobalLinkageKindInternalLinkage;
  case cir::GlobalLinkageKind::PrivateLinkage:
    return MlirCIRGlobalLinkageKindPrivateLinkage;
  case cir::GlobalLinkageKind::ExternalWeakLinkage:
    return MlirCIRGlobalLinkageKindExternalWeakLinkage;
  case cir::GlobalLinkageKind::CommonLinkage:
    return MlirCIRGlobalLinkageKindCommonLinkage;
  }
  llvm_unreachable("Unknown GlobalLinkageKind");
}

//===----------------------------------------------------------------------===//
// CIR CallingConv
//===----------------------------------------------------------------------===//

MlirAttribute mlirCIRCallingConvAttrGet(MlirContext ctx,
                                        MlirCIRCallingConv conv) {
  cir::CallingConv cirConv;
  switch (conv) {
  case MlirCIRCallingConvC:
    cirConv = cir::CallingConv::C;
    break;
  case MlirCIRCallingConvSpirKernel:
    cirConv = cir::CallingConv::SpirKernel;
    break;
  case MlirCIRCallingConvSpirFunction:
    cirConv = cir::CallingConv::SpirFunction;
    break;
  case MlirCIRCallingConvOpenCLKernel:
    cirConv = cir::CallingConv::OpenCLKernel;
    break;
  case MlirCIRCallingConvPTXKernel:
    cirConv = cir::CallingConv::PTXKernel;
    break;
  }
  return wrap(cir::CallingConvAttr::get(unwrap(ctx), cirConv));
}

bool mlirAttributeIsACIRCallingConvAttr(MlirAttribute attr) {
  return llvm::isa<cir::CallingConvAttr>(unwrap(attr));
}

MlirCIRCallingConv mlirCIRCallingConvAttrGetConv(MlirAttribute attr) {
  auto convAttr = mlir::cast<cir::CallingConvAttr>(unwrap(attr));
  switch (convAttr.getValue()) {
  case cir::CallingConv::C:
    return MlirCIRCallingConvC;
  case cir::CallingConv::SpirKernel:
    return MlirCIRCallingConvSpirKernel;
  case cir::CallingConv::SpirFunction:
    return MlirCIRCallingConvSpirFunction;
  case cir::CallingConv::OpenCLKernel:
    return MlirCIRCallingConvOpenCLKernel;
  case cir::CallingConv::PTXKernel:
    return MlirCIRCallingConvPTXKernel;
  }
  llvm_unreachable("Unknown CallingConv");
}
