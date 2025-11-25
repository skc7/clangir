#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._cir_ops_gen import *
from ._cir_ops_gen import _Dialect
from .._mlir_libs import _cirDialect
from ..ir import Context, Type, Attribute

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
)

# Import all CIR native attributes
from .._mlir_libs._cirDialect import (
    IntAttr as _CIRIntAttr,
    BoolAttr as _CIRBoolAttr,
    FPAttr as _CIRFPAttr,
    ZeroAttr as _CIRZeroAttr,
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
