#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._cir_ops_gen import *
from ._cir_ops_gen import _Dialect
from .._mlir_libs import _cirDialect
from ..ir import Context, Type, Attribute

# Try to import VisibilityAttr if available in generated code
try:
    from ._cir_ops_gen import VisibilityAttr as _CIRVisibilityAttr
    _HAS_VISIBILITY_ATTR = True
except ImportError:
    _HAS_VISIBILITY_ATTR = False
    _CIRVisibilityAttr = None

# Import all CIR native types
from .._mlir_libs._cirDialect import (
    IntType as _CIRIntType,
    BoolType as _CIRBoolType,
    VoidType as _CIRVoidType,
    PointerType as _CIRPointerType,
    ArrayType as _CIRArrayType,
    FloatType as _CIRFloatType,
    DoubleType as _CIRDoubleType,
    FP16Type as _CIRFP16Type,
    BF16Type as _CIRBF16Type,
    FP80Type as _CIRFP80Type,
    FP128Type as _CIRFP128Type,
    ComplexType as _CIRComplexType,
    FuncType as _CIRFuncType,
    VectorType as _CIRVectorType,
    RecordType as _CIRRecordType,
    MethodType as _CIRMethodType,
    DataMemberType as _CIRDataMemberType,
    VPtrType as _CIRVPtrType,
    ExceptionType as _CIRExceptionType,
)

# Import all CIR native attributes
from .._mlir_libs._cirDialect import (
    IntAttr as _CIRIntAttr,
    BoolAttr as _CIRBoolAttr,
    FPAttr as _CIRFPAttr,
    ZeroAttr as _CIRZeroAttr,
    VisibilityAttr as _CIRVisibilityAttr,
    VisibilityKind as _CIRVisibilityKind,
    ExtraFuncAttributesAttr as _CIRExtraFuncAttributesAttr,
    GlobalLinkageKindAttr as _CIRGlobalLinkageKindAttr,
    GlobalLinkageKind as _CIRGlobalLinkageKind,
    CallingConvAttr as _CIRCallingConvAttr,
    CallingConv as _CIRCallingConv,
)

# Register dialect on module import
def _register_on_import():
    try:
        ctx = Context._get_live_context()
        if ctx is not None:
            _cirDialect.register_dialect(ctx, load=True)
    except:
        pass

_register_on_import()

def register_dialect(ctx=None):
    """Explicitly register the CIR dialect with a context."""
    if ctx is None:
        ctx = Context.current
    _cirDialect.register_dialect(ctx, load=True)


# ===----------------------------------------------------------------------===//
# CIR Type Checking Helpers
# ===----------------------------------------------------------------------===//

def is_int_type(ty: Type) -> bool:
    """Check if a type is a CIR integer type."""
    return _CIRIntType.isinstance(ty)


def is_bool_type(ty: Type) -> bool:
    """Check if a type is a CIR bool type."""
    return _CIRBoolType.isinstance(ty)


def is_float_type(ty: Type) -> bool:
    """Check if a type is a CIR float type (any precision)."""
    return (_CIRFloatType.isinstance(ty) or
            _CIRDoubleType.isinstance(ty) or
            _CIRFP16Type.isinstance(ty) or
            _CIRBF16Type.isinstance(ty) or
            _CIRFP80Type.isinstance(ty) or
            _CIRFP128Type.isinstance(ty))


def is_scalar_type(ty: Type) -> bool:
    """
    Check if a type is a CIR scalar type (int, float, or bool).
    
    Scalar types support arithmetic and comparison operations.
    Useful for determining if operator overloading can be applied.
    
    Args:
        ty: MLIR Type to check
    
    Returns:
        True if the type is a scalar, False otherwise
    
    Example:
        >>> t = cir.s32()
        >>> cir.is_scalar_type(t)  # True
        >>> ptr = cir.PointerType(cir.s32())
        >>> cir.is_scalar_type(ptr)  # False
    """
    return is_int_type(ty) or is_bool_type(ty) or is_float_type(ty)

# Export enums for direct use
VisibilityKind = _CIRVisibilityKind
GlobalLinkageKind = _CIRGlobalLinkageKind
CallingConv = _CIRCallingConv


# ===----------------------------------------------------------------------===//
# CIR Type Helpers
# ===----------------------------------------------------------------------===//

def IntType(width: int, is_signed: bool = True) -> Type:
    """
    Create a CIR integer type.

    Args:
        width: Bit width of the integer (1-128)
        is_signed: Whether the integer is signed (default: True)

    Returns:
        A CIR integer type (!cir.int<s|u, width>)

    Examples:
        >>> s32 = IntType(32, is_signed=True)   # !s32i or !cir.int<s, 32>
        >>> u64 = IntType(64, is_signed=False)  # !u64i or !cir.int<u, 64>
    """
    return _CIRIntType.get(width, is_signed)


def BoolType() -> Type:
    """
    Create a CIR bool type.

    Returns:
        A CIR bool type (!cir.bool)
    """
    return _CIRBoolType.get()


def PointerType(pointee: Type) -> Type:
    """
    Create a CIR pointer type.

    Args:
        pointee: The type being pointed to

    Returns:
        A CIR pointer type (!cir.ptr<pointee>)

    Example:
        >>> int_ptr = PointerType(IntType(32))  # !cir.ptr<!s32i>
    """
    return _CIRPointerType.get(pointee)


def VoidType() -> Type:
    """
    Create a CIR void type.

    Returns:
        A CIR void type (!cir.void)
    """
    return _CIRVoidType.get()


def FloatType() -> Type:
    """
    Create a CIR single-precision float type.

    Returns:
        A CIR float type (!cir.float)

    Example:
        >>> f = FloatType()  # !cir.float
    """
    return _CIRFloatType.get()


def DoubleType() -> Type:
    """
    Create a CIR double-precision float type.

    Returns:
        A CIR double type (!cir.double)

    Example:
        >>> d = DoubleType()  # !cir.double
    """
    return _CIRDoubleType.get()


def FP16Type() -> Type:
    """
    Create a CIR FP16 type (IEEE 754 binary16).

    Returns:
        A CIR f16 type (!cir.f16)
    """
    return _CIRFP16Type.get()


def BF16Type() -> Type:
    """
    Create a CIR BFloat16 type.

    Returns:
        A CIR bf16 type (!cir.bf16)
    """
    return _CIRBF16Type.get()


def FP80Type() -> Type:
    """
    Create a CIR FP80 type (x86 extended precision).

    Returns:
        A CIR f80 type (!cir.f80)
    """
    return _CIRFP80Type.get()


def FP128Type() -> Type:
    """
    Create a CIR FP128 type (IEEE 754 binary128).

    Returns:
        A CIR f128 type (!cir.f128)
    """
    return _CIRFP128Type.get()


def ComplexType(element_type: Type) -> Type:
    """
    Create a CIR complex type.

    Args:
        element_type: The type of the real and imaginary parts
                     (must be a CIR int or float type)

    Returns:
        A CIR complex type (!cir.complex<element_type>)

    Example:
        >>> c = ComplexType(FloatType())  # !cir.complex<!cir.float>
        >>> c = ComplexType(s32())        # !cir.complex<!s32i>
    """
    return _CIRComplexType.get(element_type)


def ArrayType(element_type: Type, size: int) -> Type:
    """
    Create a CIR array type (constant-size C array).

    Args:
        element_type: The type of array elements
        size: The number of elements in the array

    Returns:
        A CIR array type (!cir.array<element_type x size>)

    Example:
        >>> arr = ArrayType(s32(), 10)  # !cir.array<!s32i x 10>
        >>> arr = ArrayType(FloatType(), 5)  # !cir.array<!cir.float x 5>
    """
    return _CIRArrayType.get(element_type, size)


def FuncType(inputs: list, return_type: Type = None, is_vararg: bool = False) -> Type:
    """
    Create a CIR function type.

    Args:
        inputs: List of parameter types
        return_type: Optional return type (None for void-returning functions)
        is_vararg: Whether the function is variadic (default: False)

    Returns:
        A CIR function type (!cir.func<...>)

    Examples:
        >>> f1 = FuncType([])  # !cir.func<()>
        >>> f2 = FuncType([], BoolType())  # !cir.func<() -> !cir.bool>
        >>> f3 = FuncType([s8(), s8()])  # !cir.func<(!s8i, !s8i)>
        >>> f4 = FuncType([s8(), s8()], s32())  # !cir.func<(!s8i, !s8i) -> !s32i>
        >>> f5 = FuncType([s32()], s32(), is_vararg=True)  # !cir.func<(!s32i, ...) -> !s32i>
    """
    # Convert None return type to VoidType for native binding
    ret_type = return_type if return_type is not None else VoidType()
    return _CIRFuncType.get(inputs, ret_type, is_vararg)


def VectorType(element_type: Type, size: int) -> Type:
    """
    Create a CIR vector type.

    Args:
        element_type: The element type (must be a scalar CIR type)
        size: Number of elements (must be positive)

    Returns:
        A CIR vector type (!cir.vector<element_type x size>)

    Examples:
        >>> vec = VectorType(IntType(8, False), 4)  # !cir.vector<!u8i x 4>
        >>> vec = VectorType(FloatType(), 2)  # !cir.vector<!cir.float x 2>
    """
    return _CIRVectorType.get(element_type, size)


def RecordType(members: list, packed: bool = False, padded: bool = False,
               kind: bool = False) -> Type:
    """
    Create a CIR anonymous record type.

    Args:
        members: List of member types
        packed: Whether the record is packed (default: False)
        padded: Whether the record is padded (default: False)
        kind: Record kind - False for struct, True for class (default: False)

    Returns:
        A CIR anonymous record type (!cir.record<...>)

    Examples:
        >>> struct = RecordType([s32(), s64()])  # anonymous struct
        >>> cls = RecordType([s32()], kind=True)  # anonymous class
        >>> packed = RecordType([s8(), s32()], packed=True)  # packed struct
    """
    return _CIRRecordType.get(members, packed, padded, kind)


def MethodType(member_func_type: Type, class_type: Type) -> Type:
    """
    Create a CIR method type (pointer-to-member-function).

    Args:
        member_func_type: The function type (!cir.func<...>)
        class_type: The class type (!cir.record<...>)

    Returns:
        A CIR method type (!cir.method<func_type in class_type>)

    Example:
        >>> cls = RecordType("MyClass", [s32()])
        >>> func = FuncType([s32()], BoolType())
        >>> method = MethodType(func, cls)
    """
    return _CIRMethodType.get(member_func_type, class_type)


def DataMemberType(member_type: Type, class_type: Type) -> Type:
    """
    Create a CIR data member type (pointer-to-data-member).

    Args:
        member_type: The member type
        class_type: The class type (!cir.record<...>)

    Returns:
        A CIR data member type (!cir.data_member<member_type in class_type>)

    Example:
        >>> cls = RecordType("MyClass", [s32()])
        >>> data_member = DataMemberType(s32(), cls)
    """
    return _CIRDataMemberType.get(member_type, class_type)


def VPtrType() -> Type:
    """
    Create a CIR vptr type.

    Returns:
        A CIR vptr type (!cir.vptr)

    This type is used for the vptr member of C++ objects with virtual functions.
    """
    return _CIRVPtrType.get()


def ExceptionType() -> Type:
    """
    Create a CIR exception info type.

    Returns:
        A CIR exception info type (!cir.exception)

    This type holds information for an inflight exception.
    """
    return _CIRExceptionType.get()


# Common float type aliases
f16 = FP16Type
bf16 = BF16Type
f32 = FloatType
f64 = DoubleType
f80 = FP80Type
f128 = FP128Type


# Common integer type aliases
s8 = lambda: IntType(8, is_signed=True)
s16 = lambda: IntType(16, is_signed=True)
s32 = lambda: IntType(32, is_signed=True)
s64 = lambda: IntType(64, is_signed=True)
s128 = lambda: IntType(128, is_signed=True)

u8 = lambda: IntType(8, is_signed=False)
u16 = lambda: IntType(16, is_signed=False)
u32 = lambda: IntType(32, is_signed=False)
u64 = lambda: IntType(64, is_signed=False)
u128 = lambda: IntType(128, is_signed=False)


# ===----------------------------------------------------------------------===//
# CIR Attribute Helpers
# ===----------------------------------------------------------------------===//

def IntAttr(value: int, type: Type) -> Attribute:
    """
    Create a CIR integer attribute.

    Args:
        value: The integer value
        type: The CIR integer type

    Returns:
        A CIR integer attribute (#cir.int<value> : type)

    Example:
        >>> attr = IntAttr(42, s32())  # #cir.int<42> : !s32i
    """
    return _CIRIntAttr.get(value, type)


def BoolAttr(value: bool, type: Type = None) -> Attribute:
    """
    Create a CIR bool attribute.

    Args:
        value: The boolean value
        type: The CIR bool type (optional, defaults to !cir.bool)

    Returns:
        A CIR bool attribute (#cir.bool<true|false> : type)

    Example:
        >>> attr = BoolAttr(True)  # #cir.bool<true> : !cir.bool
    """
    if type is None:
        type = BoolType()
    return _CIRBoolAttr.get(value, type)


def FloatAttr(value: float, type: Type) -> Attribute:
    """
    Create a CIR floating-point attribute.

    Args:
        value: The floating-point value
        type: The CIR float type

    Returns:
        A CIR float attribute (#cir.fp<value> : type)

    Example:
        >>> attr = FloatAttr(3.14, FloatType())  # #cir.fp<3.140000e+00> : !cir.float
    """
    return _CIRFPAttr.get(value, type)


def ZeroAttr(type: Type) -> Attribute:
    """
    Create a CIR zero attribute (works for any type).

    Args:
        type: The CIR type

    Returns:
        A CIR zero attribute (#cir.zero : type)

    Example:
        >>> attr = ZeroAttr(s32())  # #cir.zero : !s32i
        >>> attr = ZeroAttr(ArrayType(s32(), 10))  # #cir.zero : !cir.array<!s32i x 10>
    """
    return _CIRZeroAttr.get(type)


# ===----------------------------------------------------------------------===//
# CIR FuncOp Helper Attributes
# ===----------------------------------------------------------------------===//

def VisibilityAttr(visibility = "default") -> Attribute:
    """
    Create a CIR visibility attribute.
    
    Args:
        visibility: Visibility level - can be:
                   - String: "default", "hidden", or "protected"
                   - VisibilityKind enum value
                   - Integer: 0 (default), 1 (hidden), 2 (protected)
    
    Returns:
        A CIR visibility attribute (#cir.visibility<...>)
    
    Example:
        >>> vis = VisibilityAttr("default")
        >>> vis = VisibilityAttr("hidden")
        >>> vis = VisibilityAttr(_CIRVisibilityKind.Hidden)
        >>> vis = VisibilityAttr(1)  # 1 = hidden
    
    Note:
        This attribute is used for specifying symbol visibility in CIR,
        such as for global functions and variables.
    """
    kind = None
    
    # Handle different input types
    if isinstance(visibility, str):
        visibility_map = {
            "default": _CIRVisibilityKind.Default,
            "hidden": _CIRVisibilityKind.Hidden,
            "protected": _CIRVisibilityKind.Protected
        }
        if visibility.lower() not in visibility_map:
            raise ValueError(
                f"Invalid visibility '{visibility}'. "
                f"Must be one of: 'default', 'hidden', 'protected'"
            )
        kind = visibility_map[visibility.lower()]
    elif isinstance(visibility, int):
        # Map integer to enum
        int_map = {
            0: _CIRVisibilityKind.Default,
            1: _CIRVisibilityKind.Hidden,
            2: _CIRVisibilityKind.Protected
        }
        if visibility not in int_map:
            raise ValueError(
                f"Invalid visibility {visibility}. "
                f"Must be 0 (default), 1 (hidden), or 2 (protected)"
            )
        kind = int_map[visibility]
    elif isinstance(visibility, _CIRVisibilityKind):
        # Already a VisibilityKind enum
        kind = visibility
    else:
        raise TypeError(
            f"visibility must be str, int, or VisibilityKind, got {type(visibility)}"
        )
    
    return _CIRVisibilityAttr.get(kind)

def ExtraFuncAttr(**kwargs) -> Attribute:
    """
    Create a CIR extra function attributes attribute.
    
    Args:
        **kwargs: Keyword arguments where keys are attribute names and values are MLIR Attributes
    
    Returns:
        A CIR ExtraFuncAttributesAttr (#cir.extra(...))
    
    Examples:
        >>> # Empty extra attributes
        >>> extra = ExtraFuncAttr()  # Returns #cir.extra({})
        
        >>> # With custom attributes
        >>> from mlir.ir import StringAttr, UnitAttr
        >>> extra = ExtraFuncAttr(
        ...     inline=UnitAttr.get(),
        ...     custom_attr=StringAttr.get("value")
        ... )
    
    Note:
        This is used for cir.FuncOp's extra_attrs parameter.
        The dictionary can contain any additional attributes you want to attach
        to the function operation.
    """
    from mlir.ir import DictAttr
    dict_attr = DictAttr.get(kwargs if kwargs else {})
    return _CIRExtraFuncAttributesAttr.get(dict_attr)


def GlobalLinkageKindAttr(linkage="external") -> Attribute:
    """
    Create a CIR global linkage kind attribute.
    
    Args:
        linkage: Linkage kind - can be:
                 - String: "external", "internal", "weak", etc.
                 - GlobalLinkageKind enum value
    
    Returns:
        A CIR GlobalLinkageKindAttr (#cir.linkage<...>)
    
    Example:
        >>> linkage = GlobalLinkageKindAttr("internal")
        >>> linkage = GlobalLinkageKindAttr(GlobalLinkageKind.InternalLinkage)
    """
    if isinstance(linkage, str):
        linkage_map = {
            "external": _CIRGlobalLinkageKind.ExternalLinkage,
            "available_externally": _CIRGlobalLinkageKind.AvailableExternallyLinkage,
            "linkonce": _CIRGlobalLinkageKind.LinkOnceAnyLinkage,
            "linkonce_odr": _CIRGlobalLinkageKind.LinkOnceODRLinkage,
            "weak": _CIRGlobalLinkageKind.WeakAnyLinkage,
            "weak_odr": _CIRGlobalLinkageKind.WeakODRLinkage,
            "internal": _CIRGlobalLinkageKind.InternalLinkage,
            "private": _CIRGlobalLinkageKind.PrivateLinkage,
            "extern_weak": _CIRGlobalLinkageKind.ExternalWeakLinkage,
            "common": _CIRGlobalLinkageKind.CommonLinkage,
        }
        if linkage.lower() not in linkage_map:
            raise ValueError(f"Invalid linkage '{linkage}'")
        kind = linkage_map[linkage.lower()]
    elif isinstance(linkage, _CIRGlobalLinkageKind):
        kind = linkage
    else:
        raise TypeError(f"linkage must be str or GlobalLinkageKind, got {type(linkage)}")
    
    return _CIRGlobalLinkageKindAttr.get(kind)


def CallingConvAttr(conv="c") -> Attribute:
    """
    Create a CIR calling convention attribute.
    
    Args:
        conv: Calling convention - can be:
              - String: "c", "spir_kernel", "spir_function", etc.
              - CallingConv enum value
    
    Returns:
        A CIR CallingConvAttr (#cir.calling_conv<...>)
    
    Example:
        >>> cc = CallingConvAttr("c")
        >>> cc = CallingConvAttr(CallingConv.C)
    """
    if isinstance(conv, str):
        conv_map = {
            "c": _CIRCallingConv.C,
            "spir_kernel": _CIRCallingConv.SpirKernel,
            "spir_function": _CIRCallingConv.SpirFunction,
            "opencl_kernel": _CIRCallingConv.OpenCLKernel,
            "ptx_kernel": _CIRCallingConv.PTXKernel,
        }
        if conv.lower() not in conv_map:
            raise ValueError(f"Invalid calling convention '{conv}'")
        calling_conv = conv_map[conv.lower()]
    elif isinstance(conv, _CIRCallingConv):
        calling_conv = conv
    else:
        raise TypeError(f"conv must be str or CallingConv, got {type(conv)}")
    
    return _CIRCallingConvAttr.get(calling_conv)
