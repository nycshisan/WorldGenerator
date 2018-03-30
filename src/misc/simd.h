//
// Created by Nycshisan on 2018/3/30.
//

#ifndef WORLDGENERATOR_SIMD_H
#define WORLDGENERATOR_SIMD_H

/*
 * SIMD vectors and matrices arithmetic
 */

#include <cassert>
#include <x86intrin.h>

typedef float data_t;

#define ALWAYS_INLINE __attribute__((always_inline))

/*
 * Here are vector definitions.
 * Always use 128-bit memory to store the vector regardless of the size of the vector
 * to ensure accordance in the vector arithmetic.
 */

#define DEF_VEC(SIZE) \
typedef union { \
    __m128 m; \
    data_t data[SIZE]; \
    struct { \
        data_t x, y, z, w; \
    }; \
    struct { \
        data_t r, g, b, a; \
    }; \
    ALWAYS_INLINE data_t& operator [] (size_t i) { \
        assert(i < SIZE); \
        return this->data[i]; \
    } \
    ALWAYS_INLINE data_t operator [] (size_t i) const { \
        assert(i < SIZE); \
        return this->data[i]; \
    } \
} vec##SIZE __attribute__((aligned(16)))

DEF_VEC(4);
DEF_VEC(3);
DEF_VEC(2);

#undef DEF_VEC

// Vector arithmetic
#define DECL_VEC_PLUSMINUS(SIZE, OPER, FUNC_NAME) \
ALWAYS_INLINE vec##SIZE operator OPER (const vec##SIZE &a, const vec##SIZE &b); \
ALWAYS_INLINE void operator OPER##= (vec##SIZE &a, const vec##SIZE &b);

DECL_VEC_PLUSMINUS(2, +, add)
DECL_VEC_PLUSMINUS(3, +, add)
DECL_VEC_PLUSMINUS(4, +, add)

DECL_VEC_PLUSMINUS(2, -, sub)
DECL_VEC_PLUSMINUS(3, -, sub)
DECL_VEC_PLUSMINUS(4, -, sub)

#undef DECL_VEC_PLUSMINUS

#define DECL_VEC_MULDIV(SIZE, OPER, FUNC_NAME) \
ALWAYS_INLINE vec##SIZE operator OPER (const vec##SIZE &v, data_t scalar); \
ALWAYS_INLINE void operator OPER##= (vec##SIZE &v, data_t scalar);

DECL_VEC_MULDIV(2, *, mul)
DECL_VEC_MULDIV(3, *, mul)
DECL_VEC_MULDIV(4, *, mul)

DECL_VEC_MULDIV(2, /, div)
DECL_VEC_MULDIV(3, /, div)
DECL_VEC_MULDIV(4, /, div)

#undef DECL_VEC_MULDIV

// Vector dot product
#define DOT_CONTROL_2 0b00111111
#define DOT_CONTROL_3 0b01111111
#define DOT_CONTROL_4 0b11111111

#define DECL_VEC_DOT(SIZE) \
ALWAYS_INLINE __m128 __dot(const vec##SIZE &a, const vec##SIZE &b); \
ALWAYS_INLINE data_t dot(const vec##SIZE &a, const vec##SIZE &b);

DECL_VEC_DOT(2)
DECL_VEC_DOT(3)
DECL_VEC_DOT(4)

#undef DOT_CONTROL_2
#undef DOT_CONTROL_3
#undef DOT_CONTROL_4

#undef DECL_VEC_DOT

/*
 * Here are matrix definitions.
 * Always use 512-bit memory to store the martix regardless of the size of the matrix
 * to ensure accordance in the martix arithmetics.
 */

#define DEF_MAT(SIZE) \
typedef union { \
    __m128 col[4]; \
    struct { \
        data_t _11, _12, _13, _14, \
               _21, _22, _23, _24, \
               _31, _32, _33, _34, \
               _41, _42, _43, _44; \
    }; \
    ALWAYS_INLINE __m128& operator [] (size_t i) { \
        assert(i < SIZE); \
        return this->col[i]; \
    } \
    ALWAYS_INLINE __m128 operator [] (size_t i) const { \
        assert(i < SIZE); \
        return this->col[i]; \
    } \
} mat##SIZE __attribute__((aligned(16)))

DEF_MAT(2);
DEF_MAT(3);
DEF_MAT(4);

#undef DEF_MAT

// Matrix arithmetic
#define DECL_MAT_PLUSMINUS(SIZE, OPER, FUNC_NAME) \
ALWAYS_INLINE mat##SIZE operator OPER (const mat##SIZE &a, const mat##SIZE &b); \
ALWAYS_INLINE void operator OPER##= (mat##SIZE &a, const mat##SIZE &b);

DECL_MAT_PLUSMINUS(2, +, add)
DECL_MAT_PLUSMINUS(3, +, add)
DECL_MAT_PLUSMINUS(4, +, add)

DECL_MAT_PLUSMINUS(2, -, sub)
DECL_MAT_PLUSMINUS(3, -, sub)
DECL_MAT_PLUSMINUS(4, -, sub)

#undef DECL_MAT_PLUSMINUS

#define DECL_MAT_MULDIV(SIZE, OPER, FUNC_NAME) \
ALWAYS_INLINE mat##SIZE operator OPER (const mat##SIZE &v, data_t scalar); \
ALWAYS_INLINE void operator OPER##= (mat##SIZE &v, data_t scalar);

DECL_MAT_MULDIV(2, *, mul)
DECL_MAT_MULDIV(3, *, mul)
DECL_MAT_MULDIV(4, *, mul)

DECL_MAT_MULDIV(2, /, div)
DECL_MAT_MULDIV(3, /, div)
DECL_MAT_MULDIV(4, /, div)

#undef DECL_MAT_MULDIV

// Matrix dot product
#define DECL_MAT_DOT(SIZE) \
ALWAYS_INLINE vec##SIZE operator * (const mat##SIZE &m, const vec##SIZE &v); \
ALWAYS_INLINE mat##SIZE operator * (mat##SIZE &a, const mat##SIZE &b);

DECL_MAT_DOT(2)
DECL_MAT_DOT(3)
DECL_MAT_DOT(4)

#undef DECL_MAT_DOT

// Matrix transpose
// We could use the same function to transpose mat2/3/4 because we store them in an accordance 512-bit container
// t0 = {_11, _21, _12, _22}
// t1 = {_13, _23, _14, _24}
// t2 = {_31, _41, _32, _42}
// t3 = {_33, _43, _34, _44}
// col[0] = {_11, _21, _31, _41}
// col[1] = {_12, _22, _32, _42}
// col[2] = {_13, _23, _33, _43}
// col[3] = {_13, _24, _34, _44}
#define DECL_MAT_TRANSPOSE(SIZE) \
ALWAYS_INLINE void transpose(mat##SIZE &m);

DECL_MAT_TRANSPOSE(2)
DECL_MAT_TRANSPOSE(3)
DECL_MAT_TRANSPOSE(4)

#undef DECL_MAT_TRANSPOSE

// Matrix inverse
ALWAYS_INLINE void invert(mat2 &m);

ALWAYS_INLINE void invert(mat3 &m);

// Referenced: http://swiborg.com/download/dev/pdf/simd-inverse-of-4x4-matrix.pdf
ALWAYS_INLINE void invert(mat4 &m);

#endif //WORLDGENERATOR_SIMD_H
