//===---- AMDGPU.cpp - AMDGPU-specific CIR CodeGen ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides AMDGPU-specific CIR CodeGen logic for function attributes.
//
//===----------------------------------------------------------------------===//

#include "../CIRGenModule.h"
#include "../TargetInfo.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

/// Check if AMDGPU protected visibility is required.
static bool requiresAMDGPUProtectedVisibility(const clang::Decl *d,
                                              cir::FuncOp func) {
  if (func.getGlobalVisibility() != cir::VisibilityKind::Hidden)
    return false;

  if (d->hasAttr<OMPDeclareTargetDeclAttr>())
    return false;

  return d->hasAttr<DeviceKernelAttr>() ||
         (clang::isa<clang::FunctionDecl>(d) && d->hasAttr<CUDAGlobalAttr>());
}

/// Handle amdgpu-flat-work-group-size attribute.
static void handleAMDGPUFlatWorkGroupSizeAttr(const clang::FunctionDecl *fd,
                                              mlir::NamedAttrList &extraAttrs,
                                              CIRGenModule &cgm,
                                              bool isOpenCLKernel) {
  auto &builder = cgm.getBuilder();
  const auto *flatWGS = fd->getAttr<AMDGPUFlatWorkGroupSizeAttr>();
  const auto *reqdWGS =
      cgm.getLangOpts().OpenCL ? fd->getAttr<ReqdWorkGroupSizeAttr>() : nullptr;

  if (flatWGS || reqdWGS) {
    unsigned min = 0, max = 0;
    if (flatWGS) {
      min = flatWGS->getMin()
                ->EvaluateKnownConstInt(cgm.getASTContext())
                .getExtValue();
      max = flatWGS->getMax()
                ->EvaluateKnownConstInt(cgm.getASTContext())
                .getExtValue();
    }
    if (reqdWGS && min == 0 && max == 0) {
      min = max = reqdWGS->getXDim()
                      ->EvaluateKnownConstInt(cgm.getASTContext())
                      .getExtValue() *
                  reqdWGS->getYDim()
                      ->EvaluateKnownConstInt(cgm.getASTContext())
                      .getExtValue() *
                  reqdWGS->getZDim()
                      ->EvaluateKnownConstInt(cgm.getASTContext())
                      .getExtValue();
    }
    if (min != 0) {
      assert(min <= max && "Min must be less than or equal Max");
      std::string attrVal = llvm::utostr(min) + "," + llvm::utostr(max);
      extraAttrs.set(builder.getStringAttr("amdgpu-flat-work-group-size"),
                     builder.getStringAttr(attrVal));
    } else {
      assert(max == 0 && "Max must be zero");
    }
  } else {
    // Default: "1,256" for OpenCL, "1,GPUMaxThreadsPerBlock" for HIP
    const unsigned defaultMax =
        isOpenCLKernel ? 256 : cgm.getLangOpts().GPUMaxThreadsPerBlock;
    std::string attrVal = std::string("1,") + llvm::utostr(defaultMax);
    extraAttrs.set(builder.getStringAttr("amdgpu-flat-work-group-size"),
                   builder.getStringAttr(attrVal));
  }
}

/// Handle amdgpu-waves-per-eu attribute.
static void handleAMDGPUWavesPerEUAttr(const clang::FunctionDecl *fd,
                                       mlir::NamedAttrList &extraAttrs,
                                       CIRGenModule &cgm) {
  const auto *attr = fd->getAttr<AMDGPUWavesPerEUAttr>();
  if (!attr)
    return;

  auto &builder = cgm.getBuilder();
  unsigned min =
      attr->getMin()->EvaluateKnownConstInt(cgm.getASTContext()).getExtValue();
  unsigned max = attr->getMax()
                     ? attr->getMax()
                           ->EvaluateKnownConstInt(cgm.getASTContext())
                           .getExtValue()
                     : 0;

  if (min != 0) {
    assert((max == 0 || min <= max) && "Min must be less than or equal Max");
    std::string attrVal = llvm::utostr(min);
    if (max != 0)
      attrVal = attrVal + "," + llvm::utostr(max);
    extraAttrs.set(builder.getStringAttr("amdgpu-waves-per-eu"),
                   builder.getStringAttr(attrVal));
  } else {
    assert(max == 0 && "Max must be zero");
  }
}

/// Handle amdgpu-num-sgpr attribute.
static void handleAMDGPUNumSGPRAttr(const clang::FunctionDecl *fd,
                                    mlir::NamedAttrList &extraAttrs,
                                    CIRGenModule &cgm) {
  const auto *attr = fd->getAttr<AMDGPUNumSGPRAttr>();
  if (!attr)
    return;

  uint32_t numSGPR = attr->getNumSGPR();
  if (numSGPR != 0) {
    auto &builder = cgm.getBuilder();
    extraAttrs.set(builder.getStringAttr("amdgpu-num-sgpr"),
                   builder.getStringAttr(llvm::utostr(numSGPR)));
  }
}

/// Handle amdgpu-num-vgpr attribute.
static void handleAMDGPUNumVGPRAttr(const clang::FunctionDecl *fd,
                                    mlir::NamedAttrList &extraAttrs,
                                    CIRGenModule &cgm) {
  const auto *attr = fd->getAttr<AMDGPUNumVGPRAttr>();
  if (!attr)
    return;

  uint32_t numVGPR = attr->getNumVGPR();
  if (numVGPR != 0) {
    auto &builder = cgm.getBuilder();
    extraAttrs.set(builder.getStringAttr("amdgpu-num-vgpr"),
                   builder.getStringAttr(llvm::utostr(numVGPR)));
  }
}

/// Handle amdgpu-max-num-workgroups attribute.
static void handleAMDGPUMaxNumWorkGroupsAttr(const clang::FunctionDecl *fd,
                                             mlir::NamedAttrList &extraAttrs,
                                             CIRGenModule &cgm) {
  const auto *attr = fd->getAttr<AMDGPUMaxNumWorkGroupsAttr>();
  if (!attr)
    return;

  auto &builder = cgm.getBuilder();
  uint32_t x = attr->getMaxNumWorkGroupsX()
                   ->EvaluateKnownConstInt(cgm.getASTContext())
                   .getExtValue();
  uint32_t y = attr->getMaxNumWorkGroupsY()
                   ? attr->getMaxNumWorkGroupsY()
                         ->EvaluateKnownConstInt(cgm.getASTContext())
                         .getExtValue()
                   : 1;
  uint32_t z = attr->getMaxNumWorkGroupsZ()
                   ? attr->getMaxNumWorkGroupsZ()
                         ->EvaluateKnownConstInt(cgm.getASTContext())
                         .getExtValue()
                   : 1;

  llvm::SmallString<32> attrVal;
  llvm::raw_svector_ostream os(attrVal);
  os << x << ',' << y << ',' << z;
  extraAttrs.set(builder.getStringAttr("amdgpu-max-num-workgroups"),
                 builder.getStringAttr(attrVal.str()));
}

/// Handle amdgpu-cluster-dims attribute.
static void handleAMDGPUClusterDimsAttr(const clang::FunctionDecl *fd,
                                        mlir::NamedAttrList &extraAttrs,
                                        CIRGenModule &cgm,
                                        bool isOpenCLKernel) {
  auto &builder = cgm.getBuilder();

  // Handle explicit CUDAClusterDimsAttr
  if (const auto *attr = fd->getAttr<CUDAClusterDimsAttr>()) {
    auto getExprVal = [&](const Expr *e) {
      return e ? e->EvaluateKnownConstInt(cgm.getASTContext()).getExtValue()
               : 1;
    };
    unsigned x = getExprVal(attr->getX());
    unsigned y = getExprVal(attr->getY());
    unsigned z = getExprVal(attr->getZ());

    llvm::SmallString<32> attrVal;
    llvm::raw_svector_ostream os(attrVal);
    os << x << ',' << y << ',' << z;
    extraAttrs.set(builder.getStringAttr("amdgpu-cluster-dims"),
                   builder.getStringAttr(attrVal.str()));
  }

  // OpenCL doesn't support cluster feature - disable it
  const clang::TargetInfo &targetInfo = cgm.getASTContext().getTargetInfo();
  if ((isOpenCLKernel &&
       targetInfo.hasFeatureEnabled(targetInfo.getTargetOpts().FeatureMap,
                                    "clusters")) ||
      fd->hasAttr<CUDANoClusterAttr>()) {
    extraAttrs.set(builder.getStringAttr("amdgpu-cluster-dims"),
                   builder.getStringAttr("0,0,0"));
  }
}

/// Handle amdgpu-ieee attribute.
static void handleAMDGPUIEEEAttr(mlir::NamedAttrList &extraAttrs,
                                 CIRGenModule &cgm) {
  if (!cgm.getCodeGenOpts().EmitIEEENaNCompliantInsts) {
    auto &builder = cgm.getBuilder();
    extraAttrs.set(builder.getStringAttr("amdgpu-ieee"),
                   builder.getStringAttr("false"));
  }
}

} // anonymous namespace

void clang::CIRGen::setAMDGPUTargetFunctionAttributes(const clang::Decl *decl,
                                                      cir::FuncOp func,
                                                      CIRGenModule &cgm) {
  const auto *fd = clang::dyn_cast_or_null<clang::FunctionDecl>(decl);
  if (!fd)
    return;

  if (func.isDeclaration())
    return;

  // Set protected visibility for AMDGPU kernels
  if (requiresAMDGPUProtectedVisibility(decl, func)) {
    func.setGlobalVisibility(cir::VisibilityKind::Protected);
    func.setDSOLocal(true);
  }

  const bool isOpenCLKernel =
      cgm.getLangOpts().OpenCL && fd->hasAttr<DeviceKernelAttr>();
  const bool isHIPKernel =
      cgm.getLangOpts().HIP && fd->hasAttr<CUDAGlobalAttr>();

  if (!isOpenCLKernel && !isHIPKernel)
    return;

  // Set HIP kernel calling convention
  if (isHIPKernel) {
    func.setCallingConv(cir::CallingConv::AMDGPUKernel);
    func.setVisibility(mlir::SymbolTable::Visibility::Public);
    func.setLinkageAttr(cir::GlobalLinkageKindAttr::get(
        func.getContext(), cir::GlobalLinkageKind::ExternalLinkage));
  }

  mlir::NamedAttrList extraAttrs;
  if (auto existing = func.getExtraAttrs())
    extraAttrs.append(existing.getElements().getValue());

  handleAMDGPUFlatWorkGroupSizeAttr(fd, extraAttrs, cgm, isOpenCLKernel);
  handleAMDGPUWavesPerEUAttr(fd, extraAttrs, cgm);
  handleAMDGPUNumSGPRAttr(fd, extraAttrs, cgm);
  handleAMDGPUNumVGPRAttr(fd, extraAttrs, cgm);
  handleAMDGPUMaxNumWorkGroupsAttr(fd, extraAttrs, cgm);
  handleAMDGPUClusterDimsAttr(fd, extraAttrs, cgm, isOpenCLKernel);
  handleAMDGPUIEEEAttr(extraAttrs, cgm);

  auto &builder = cgm.getBuilder();
  func.setExtraAttrsAttr(
      cir::ExtraFuncAttributesAttr::get(builder.getDictionaryAttr(extraAttrs)));
}
