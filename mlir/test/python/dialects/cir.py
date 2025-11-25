# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import cir


def run(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        cir.register_dialect(ctx)
        f()
    return f


# CHECK-LABEL: TEST: testIntTypes
@run
def testIntTypes():
    # Test signed integer types
    s8_type = cir.s8()
    s16_type = cir.s16()
    s32_type = cir.s32()
    s64_type = cir.s64()
    s128_type = cir.s128()
    # CHECK: !cir.int<s, 8>
    print(s8_type)
    # CHECK: !cir.int<s, 16>
    print(s16_type)
    # CHECK: !cir.int<s, 32>
    print(s32_type)
    # CHECK: !cir.int<s, 64>
    print(s64_type)
    # CHECK: !cir.int<s, 128>
    print(s128_type)
    # Test unsigned integer types
    u8_type = cir.u8()
    u16_type = cir.u16()
    u32_type = cir.u32()
    u64_type = cir.u64()
    u128_type = cir.u128()
    # CHECK: !cir.int<u, 8>
    print(u8_type)
    # CHECK: !cir.int<u, 16>
    print(u16_type)
    # CHECK: !cir.int<u, 32>
    print(u32_type)
    # CHECK: !cir.int<u, 64>
    print(u64_type)
    # CHECK: !cir.int<u, 128>
    print(u128_type)
    # Test custom width integers
    custom_signed = cir.IntType(13, is_signed=True)
    custom_unsigned = cir.IntType(13, is_signed=False)
    # CHECK: !cir.int<s, 13>
    print(custom_signed)
    # CHECK: !cir.int<u, 13>
    print(custom_unsigned)


# CHECK-LABEL: TEST: testBoolType
@run
def testBoolType():
    bool_type = cir.BoolType()
    # CHECK: !cir.bool
    print(bool_type)


# CHECK-LABEL: TEST: testVoidType
@run
def testVoidType():
    void_type = cir.VoidType()
    # CHECK: !cir.void
    print(void_type)


# CHECK-LABEL: TEST: testFloatTypes
@run
def testFloatTypes():
    # Test standard float types
    f16_type = cir.f16()
    bf16_type = cir.bf16()
    f32_type = cir.f32()
    f64_type = cir.f64()
    f80_type = cir.f80()
    f128_type = cir.f128()
    # CHECK: !cir.f16
    print(f16_type)
    # CHECK: !cir.bf16
    print(bf16_type)
    # CHECK: !cir.float
    print(f32_type)
    # CHECK: !cir.double
    print(f64_type)
    # CHECK: !cir.f80
    print(f80_type)
    # CHECK: !cir.f128
    print(f128_type)


# CHECK-LABEL: TEST: testPointerTypes
@run
def testPointerTypes():
    # Pointer to int
    int_ptr = cir.PointerType(cir.s32())
    # CHECK: !cir.ptr<!cir.int<s, 32>>
    print(int_ptr)
    # Pointer to float
    float_ptr = cir.PointerType(cir.f32())
    # CHECK: !cir.ptr<!cir.float>
    print(float_ptr)
    # Pointer to void
    void_ptr = cir.PointerType(cir.VoidType())
    # CHECK: !cir.ptr<!cir.void>
    print(void_ptr)
    # Pointer to pointer
    ptr_ptr = cir.PointerType(cir.PointerType(cir.s32()))
    # CHECK: !cir.ptr<!cir.ptr<!cir.int<s, 32>>>
    print(ptr_ptr)


# CHECK-LABEL: TEST: testArrayTypes
@run
def testArrayTypes():
    # Array of integers
    int_array = cir.ArrayType(cir.s32(), 10)
    # CHECK: !cir.array<!cir.int<s, 32> x 10>
    print(int_array)
    # Array of floats
    float_array = cir.ArrayType(cir.f32(), 5)
    # CHECK: !cir.array<!cir.float x 5>
    print(float_array)
    # Array of bools
    bool_array = cir.ArrayType(cir.BoolType(), 8)
    # CHECK: !cir.array<!cir.bool x 8>
    print(bool_array)
    # Array of pointers
    ptr_array = cir.ArrayType(cir.PointerType(cir.s32()), 3)
    # CHECK: !cir.array<!cir.ptr<!cir.int<s, 32>> x 3>
    print(ptr_array)
    # Multi-dimensional array (array of arrays)
    array_2d = cir.ArrayType(cir.ArrayType(cir.s32(), 5), 3)
    # CHECK: !cir.array<!cir.array<!cir.int<s, 32> x 5> x 3>
    print(array_2d)


# CHECK-LABEL: TEST: testComplexTypes
@run
def testComplexTypes():
    # Complex float
    complex_float = cir.ComplexType(cir.f32())
    # CHECK: !cir.complex<!cir.float>
    print(complex_float)
    # Complex double
    complex_double = cir.ComplexType(cir.f64())
    # CHECK: !cir.complex<!cir.double>
    print(complex_double)
    # Complex integer
    complex_int = cir.ComplexType(cir.s32())
    # CHECK: !cir.complex<!cir.int<s, 32>>
    print(complex_int)


# CHECK-LABEL: TEST: testFuncTypes
@run
def testFuncTypes():
    # Function with no parameters and no return
    func_void = cir.FuncType([])
    # CHECK: !cir.func<()>
    print(func_void)
    # Function with no parameters, returns bool
    func_returns_bool = cir.FuncType([], cir.BoolType())
    # CHECK: !cir.func<() -> !cir.bool>
    print(func_returns_bool)
    # Function with parameters, no return
    func_with_params = cir.FuncType([cir.s32(), cir.s32()])
    # CHECK: !cir.func<(!cir.int<s, 32>, !cir.int<s, 32>)>
    print(func_with_params)
    # Function with parameters and return type
    func_full = cir.FuncType([cir.s32(), cir.f32()], cir.s64())
    # CHECK: !cir.func<(!cir.int<s, 32>, !cir.float) -> !cir.int<s, 64>>
    print(func_full)
    # Variadic function
    func_vararg = cir.FuncType([cir.s32()], cir.s32(), is_vararg=True)
    # CHECK: !cir.func<(!cir.int<s, 32>, ...) -> !cir.int<s, 32>>
    print(func_vararg)
    # Variadic function with no fixed parameters
    func_vararg_only = cir.FuncType([], cir.s32(), is_vararg=True)
    # CHECK: !cir.func<(...) -> !cir.int<s, 32>>
    print(func_vararg_only)
    # Function taking pointers
    func_with_ptrs = cir.FuncType(
        [cir.PointerType(cir.s8()), cir.PointerType(cir.s8())],
        cir.s32()
    )
    # CHECK: !cir.func<(!cir.ptr<!cir.int<s, 8>>, !cir.ptr<!cir.int<s, 8>>) -> !cir.int<s, 32>>
    print(func_with_ptrs)
    # Function pointer type
    func_ptr = cir.PointerType(cir.FuncType([cir.s32()], cir.s32()))
    # CHECK: !cir.ptr<!cir.func<(!cir.int<s, 32>) -> !cir.int<s, 32>>>
    print(func_ptr)


# CHECK-LABEL: TEST: testIntAttributes
@run
def testIntAttributes():
    # Signed integer attributes
    int_attr_s32 = cir.IntAttr(42, cir.s32())
    # CHECK: #cir.int<42> : !cir.int<s, 32>
    print(int_attr_s32)
    int_attr_s64 = cir.IntAttr(-100, cir.s64())
    # CHECK: #cir.int<-100> : !cir.int<s, 64>
    print(int_attr_s64)
    # Unsigned integer attributes
    int_attr_u8 = cir.IntAttr(255, cir.u8())
    # CHECK: #cir.int<255> : !cir.int<u, 8>
    print(int_attr_u8)
    # Zero
    int_attr_zero = cir.IntAttr(0, cir.s32())
    # CHECK: #cir.int<0> : !cir.int<s, 32>
    print(int_attr_zero)


# CHECK-LABEL: TEST: testBoolAttributes
@run
def testBoolAttributes():
    # True attribute
    bool_true = cir.BoolAttr(True)
    # CHECK: #cir.bool<true> : !cir.bool
    print(bool_true)
    # False attribute
    bool_false = cir.BoolAttr(False)
    # CHECK: #cir.bool<false> : !cir.bool
    print(bool_false)
    # With explicit type
    bool_explicit = cir.BoolAttr(True, cir.BoolType())
    # CHECK: #cir.bool<true> : !cir.bool
    print(bool_explicit)


# CHECK-LABEL: TEST: testFloatAttributes
@run
def testFloatAttributes():
    # Float attribute
    float_attr = cir.FloatAttr(3.14, cir.f32())
    # CHECK: #cir.fp<3.140000e+00> : !cir.float
    print(float_attr)
    # Double attribute
    double_attr = cir.FloatAttr(2.71828, cir.f64())
    # CHECK: #cir.fp<2.718280e+00> : !cir.double
    print(double_attr)
    # Zero float
    zero_float = cir.FloatAttr(0.0, cir.f32())
    # CHECK: #cir.fp<0.000000e+00> : !cir.float
    print(zero_float)
    # Negative float
    neg_float = cir.FloatAttr(-1.5, cir.f32())
    # CHECK: #cir.fp<-1.500000e+00> : !cir.float
    print(neg_float)


# CHECK-LABEL: TEST: testZeroAttributes
@run
def testZeroAttributes():
    # Zero for int
    zero_int = cir.ZeroAttr(cir.s32())
    # CHECK: #cir.zero : !cir.int<s, 32>
    print(zero_int)
    # Zero for float
    zero_float = cir.ZeroAttr(cir.f32())
    # CHECK: #cir.zero : !cir.float
    print(zero_float)
    # Zero for array
    zero_array = cir.ZeroAttr(cir.ArrayType(cir.s32(), 10))
    # CHECK: #cir.zero : !cir.array<!cir.int<s, 32> x 10>
    print(zero_array)
    # Zero for pointer
    zero_ptr = cir.ZeroAttr(cir.PointerType(cir.s32()))
    # CHECK: #cir.zero : !cir.ptr<!cir.int<s, 32>>
    print(zero_ptr)
    # Zero for complex
    zero_complex = cir.ZeroAttr(cir.ComplexType(cir.f64()))
    # CHECK: #cir.zero : !cir.complex<!cir.double>
    print(zero_complex)


# CHECK-LABEL: TEST: testCompositeTypes
@run
def testCompositeTypes():
    # Complex pointer types
    ptr_to_array = cir.PointerType(cir.ArrayType(cir.s32(), 5))
    # CHECK: !cir.ptr<!cir.array<!cir.int<s, 32> x 5>>
    print(ptr_to_array)
    # Array of pointers to functions
    func_type = cir.FuncType([cir.s32()], cir.s32())
    func_ptr_type = cir.PointerType(func_type)
    array_of_func_ptrs = cir.ArrayType(func_ptr_type, 3)
    # CHECK: !cir.array<!cir.ptr<!cir.func<(!cir.int<s, 32>) -> !cir.int<s, 32>>> x 3>
    print(array_of_func_ptrs)
    # Pointer to complex type
    ptr_to_complex = cir.PointerType(cir.ComplexType(cir.f64()))
    # CHECK: !cir.ptr<!cir.complex<!cir.double>>
    print(ptr_to_complex)
    # Function returning pointer
    func_ret_ptr = cir.FuncType([cir.s32()], cir.PointerType(cir.s8()))
    # CHECK: !cir.func<(!cir.int<s, 32>) -> !cir.ptr<!cir.int<s, 8>>>
    print(func_ret_ptr)


# CHECK-LABEL: TEST: testTypeAliases
@run
def testTypeAliases():
    # Test that aliases work correctly
    # Signed int aliases
    type1 = cir.s8()
    type2 = cir.IntType(8, is_signed=True)
    # CHECK: !cir.int<s, 8>
    print(type1)
    # CHECK: !cir.int<s, 8>
    print(type2)
    # Float aliases
    f32_alias = cir.f32()
    f32_full = cir.FloatType()
    # CHECK: !cir.float
    print(f32_alias)
    # CHECK: !cir.float
    print(f32_full)
    # Double aliases
    f64_alias = cir.f64()
    f64_full = cir.DoubleType()
    # CHECK: !cir.double
    print(f64_alias)
    # CHECK: !cir.double
    print(f64_full)
