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

//===----------------------------------------------------------------------===//
// CIR Type API
//===----------------------------------------------------------------------===//

/// Creates a CIR integer type.
MLIR_CAPI_EXPORTED MlirType mlirCIRIntTypeGet(MlirContext ctx, unsigned width,
                                              bool isSigned);

/// Checks if the given type is a CIR integer type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRIntType(MlirType type);

/// Creates a CIR bool type.
MLIR_CAPI_EXPORTED MlirType mlirCIRBoolTypeGet(MlirContext ctx);

/// Checks if the given type is a CIR bool type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRBoolType(MlirType type);

/// Creates a CIR void type.
MLIR_CAPI_EXPORTED MlirType mlirCIRVoidTypeGet(MlirContext ctx);

/// Checks if the given type is a CIR void type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRVoidType(MlirType type);

/// Creates a CIR pointer type.
MLIR_CAPI_EXPORTED MlirType mlirCIRPointerTypeGet(MlirType pointee);

/// Checks if the given type is a CIR pointer type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRPointerType(MlirType type);

/// Creates a CIR array type.
MLIR_CAPI_EXPORTED MlirType mlirCIRArrayTypeGet(MlirType elementType,
                                                uint64_t size);

/// Checks if the given type is a CIR array type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRArrayType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Float Types
//===----------------------------------------------------------------------===//

/// Creates a CIR single-precision float type (!cir.float).
MLIR_CAPI_EXPORTED MlirType mlirCIRFloatTypeGet(MlirContext ctx);

/// Creates a CIR double-precision float type (!cir.double).
MLIR_CAPI_EXPORTED MlirType mlirCIRDoubleTypeGet(MlirContext ctx);

/// Creates a CIR FP16 type (!cir.f16).
MLIR_CAPI_EXPORTED MlirType mlirCIRFP16TypeGet(MlirContext ctx);

/// Creates a CIR BFloat16 type (!cir.bf16).
MLIR_CAPI_EXPORTED MlirType mlirCIRBF16TypeGet(MlirContext ctx);

/// Creates a CIR FP80 type (!cir.f80).
MLIR_CAPI_EXPORTED MlirType mlirCIRFP80TypeGet(MlirContext ctx);

/// Creates a CIR FP128 type (!cir.f128).
MLIR_CAPI_EXPORTED MlirType mlirCIRFP128TypeGet(MlirContext ctx);

/// Checks if the given type is a CIR single-precision float type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRFloatType(MlirType type);

/// Checks if the given type is a CIR double-precision float type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRDoubleType(MlirType type);

/// Checks if the given type is a CIR FP16 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRFP16Type(MlirType type);

/// Checks if the given type is a CIR BFloat16 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRBF16Type(MlirType type);

/// Checks if the given type is a CIR FP80 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRFP80Type(MlirType type);

/// Checks if the given type is a CIR FP128 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRFP128Type(MlirType type);

/// Checks if the given type is a CIR long double type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRLongDoubleType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Complex Type
//===----------------------------------------------------------------------===//

/// Creates a CIR complex type.
MLIR_CAPI_EXPORTED MlirType mlirCIRComplexTypeGet(MlirType elementType);

/// Checks if the given type is a CIR complex type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRComplexType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Function Type
//===----------------------------------------------------------------------===//

/// Creates a CIR function type.
MLIR_CAPI_EXPORTED MlirType mlirCIRFuncTypeGet(MlirContext ctx,
                                               intptr_t numInputs,
                                               MlirType const *inputs,
                                               MlirType returnType,
                                               bool isVarArg);

/// Checks if the given type is a CIR function type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRFuncType(MlirType type);

/// Gets the number of input types in a CIR function type.
MLIR_CAPI_EXPORTED intptr_t mlirCIRFuncTypeGetNumInputs(MlirType type);

/// Gets the input type at the given index in a CIR function type.
MLIR_CAPI_EXPORTED MlirType mlirCIRFuncTypeGetInput(MlirType type,
                                                    intptr_t pos);

/// Gets the return type of a CIR function type.
MLIR_CAPI_EXPORTED MlirType mlirCIRFuncTypeGetReturnType(MlirType type);

/// Checks if a CIR function type is variadic.
MLIR_CAPI_EXPORTED bool mlirCIRFuncTypeIsVarArg(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Vector Type
//===----------------------------------------------------------------------===//

/// Creates a CIR vector type.
MLIR_CAPI_EXPORTED MlirType mlirCIRVectorTypeGet(MlirType elementType,
                                                 uint64_t size);

/// Checks if the given type is a CIR vector type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRVectorType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Record Type
//===----------------------------------------------------------------------===//

/// Creates a CIR identified and complete record type.
MLIR_CAPI_EXPORTED MlirType mlirCIRRecordTypeGet(MlirContext ctx,
                                                 intptr_t numMembers,
                                                 MlirType const *members,
                                                 bool packed, bool padded,
                                                 bool kind);

/// Checks if the given type is a CIR record type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRRecordType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Method Type
//===----------------------------------------------------------------------===//

/// Creates a CIR method type (pointer-to-member-function).
MLIR_CAPI_EXPORTED MlirType mlirCIRMethodTypeGet(MlirType memberFuncTy,
                                                 MlirType clsTy);

/// Checks if the given type is a CIR method type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRMethodType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR DataMember Type
//===----------------------------------------------------------------------===//

/// Creates a CIR data member type (pointer-to-data-member).
MLIR_CAPI_EXPORTED MlirType mlirCIRDataMemberTypeGet(MlirType memberTy,
                                                     MlirType clsTy);

/// Checks if the given type is a CIR data member type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRDataMemberType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR VPtr Type
//===----------------------------------------------------------------------===//

/// Creates a CIR vptr type.
MLIR_CAPI_EXPORTED MlirType mlirCIRVPtrTypeGet(MlirContext ctx);

/// Checks if the given type is a CIR vptr type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRVPtrType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Exception Type
//===----------------------------------------------------------------------===//

/// Creates a CIR exception info type.
MLIR_CAPI_EXPORTED MlirType mlirCIRExceptionTypeGet(MlirContext ctx);

/// Checks if the given type is a CIR exception type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACIRExceptionType(MlirType type);

//===----------------------------------------------------------------------===//
// CIR Attribute API
//===----------------------------------------------------------------------===//

/// Creates a CIR integer attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirCIRIntAttrGet(MlirType type,
                                                   int64_t value);

/// Checks if the given attribute is a CIR integer attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACIRIntAttr(MlirAttribute attr);

/// Creates a CIR bool attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirCIRBoolAttrGet(MlirType type, bool value);

/// Checks if the given attribute is a CIR bool attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACIRBoolAttr(MlirAttribute attr);

/// Creates a CIR floating-point attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirCIRFPAttrGet(MlirType type, double value);

/// Checks if the given attribute is a CIR floating-point attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACIRFPAttr(MlirAttribute attr);

/// Creates a CIR zero attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirCIRZeroAttrGet(MlirType type);

/// Checks if the given attribute is a CIR zero attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACIRZeroAttr(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// CIR VisibilityAttr
//===----------------------------------------------------------------------===//

/// CIR visibility kind enumeration.
typedef enum {
  MlirCIRVisibilityKindDefault = 0,
  MlirCIRVisibilityKindHidden = 1,
  MlirCIRVisibilityKindProtected = 2
} MlirCIRVisibilityKind;

/// Creates a CIR visibility attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCIRVisibilityAttrGet(MlirContext ctx, MlirCIRVisibilityKind kind);

/// Checks if the given attribute is a CIR visibility attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACIRVisibilityAttr(MlirAttribute attr);

/// Gets the visibility kind from a CIR visibility attribute.
MLIR_CAPI_EXPORTED MlirCIRVisibilityKind
mlirCIRVisibilityAttrGetKind(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// CIR ExtraFuncAttributesAttr
//===----------------------------------------------------------------------===//

/// Creates a CIR extra function attributes attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCIRExtraFuncAttributesAttrGet(MlirAttribute dictAttr);

/// Checks if the given attribute is a CIR extra function attributes attribute.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACIRExtraFuncAttributesAttr(MlirAttribute attr);

/// Gets the dictionary attribute from a CIR extra function attributes
/// attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCIRExtraFuncAttributesAttrGetElements(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// CIR GlobalLinkageKind
//===----------------------------------------------------------------------===//

/// CIR global linkage kind enumeration.
typedef enum {
  MlirCIRGlobalLinkageKindExternalLinkage = 0,
  MlirCIRGlobalLinkageKindAvailableExternallyLinkage = 1,
  MlirCIRGlobalLinkageKindLinkOnceAnyLinkage = 2,
  MlirCIRGlobalLinkageKindLinkOnceODRLinkage = 3,
  MlirCIRGlobalLinkageKindWeakAnyLinkage = 4,
  MlirCIRGlobalLinkageKindWeakODRLinkage = 5,
  MlirCIRGlobalLinkageKindInternalLinkage = 7,
  MlirCIRGlobalLinkageKindPrivateLinkage = 8,
  MlirCIRGlobalLinkageKindExternalWeakLinkage = 9,
  MlirCIRGlobalLinkageKindCommonLinkage = 10
} MlirCIRGlobalLinkageKind;

/// Creates a CIR global linkage kind attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCIRGlobalLinkageKindAttrGet(MlirContext ctx, MlirCIRGlobalLinkageKind kind);

/// Checks if the given attribute is a CIR global linkage kind attribute.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACIRGlobalLinkageKindAttr(MlirAttribute attr);

/// Gets the global linkage kind from a CIR global linkage kind attribute.
MLIR_CAPI_EXPORTED MlirCIRGlobalLinkageKind
mlirCIRGlobalLinkageKindAttrGetKind(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// CIR CallingConv
//===----------------------------------------------------------------------===//

/// CIR calling convention enumeration.
typedef enum {
  MlirCIRCallingConvC = 1,
  MlirCIRCallingConvSpirKernel = 2,
  MlirCIRCallingConvSpirFunction = 3,
  MlirCIRCallingConvOpenCLKernel = 4,
  MlirCIRCallingConvPTXKernel = 5
} MlirCIRCallingConv;

/// Creates a CIR calling convention attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCIRCallingConvAttrGet(MlirContext ctx, MlirCIRCallingConv conv);

/// Checks if the given attribute is a CIR calling convention attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACIRCallingConvAttr(MlirAttribute attr);

/// Gets the calling convention from a CIR calling convention attribute.
MLIR_CAPI_EXPORTED MlirCIRCallingConv
mlirCIRCallingConvAttrGetConv(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CLANG_C_CIR_DIALECT_CIR_H
