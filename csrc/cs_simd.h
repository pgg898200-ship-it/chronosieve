#ifndef CS_SIMD_H
#define CS_SIMD_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* =========================================================================
 * SIMD Backend Selection (NEON on arm64, AVX2 on x86_64, Scalar fallback)
 * Define CS_FORCE_SCALAR=1 to compile pure portable scalar path for parity tests.
 * ========================================================================= */

#if !defined(CS_FORCE_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
  #define CS_SIMD_NEON 1
  #define CS_SIMD_WIDTH 2
  #define CS_SIMD_NAME "NEON"
  #include <arm_neon.h>
#elif !defined(CS_FORCE_SCALAR) && defined(__AVX2__)
  #define CS_SIMD_AVX2 1
  #define CS_SIMD_WIDTH 4
  #define CS_SIMD_NAME "AVX2"
  #include <immintrin.h>
#else
  #define CS_SIMD_SCALAR 1
  #define CS_SIMD_WIDTH 1
  #define CS_SIMD_NAME "SCALAR"
#endif

/* =========================================================================
 * Shared High-Precision Math Constants
 * ========================================================================= */

#define CS_INV_LN2      1.44269504088896340736
#define CS_LN2_HI       0.693147180369123816490
#define CS_LN2_LO       1.90821492927058770002e-10
#define CS_POW2_MAGIC   (6755399441055744.0 + 1023.0) /* 1.5 * 2^52 + 1023 */

#define CS_PI_HALF_HI   1.57079632679489655800
#define CS_PI_HALF_LO   6.12323399573676603587e-17
#define CS_PI_QUARTER   0.78539816339744830962
#define CS_TAN_MAX_ARG  (1.57079632679489655800 - 1.0e-4)

/* Degree-12 Taylor coefficients 1/k! for exp(r), |r| <= ln(2)/2 (max rel err < 5e-16) */
#define CS_EXP_C0   1.0
#define CS_EXP_C1   1.0
#define CS_EXP_C2   0.5
#define CS_EXP_C3   1.66666666666666657415e-1
#define CS_EXP_C4   4.16666666666666643537e-2
#define CS_EXP_C5   8.33333333333333321769e-3
#define CS_EXP_C6   1.38888888888888894190e-3
#define CS_EXP_C7   1.98412698412698412528e-4
#define CS_EXP_C8   2.48015873015873015660e-5
#define CS_EXP_C9   2.75573192239858925107e-6
#define CS_EXP_C10  2.75573192239858882745e-7
#define CS_EXP_C11  2.50521083854417202248e-8
#define CS_EXP_C12  2.08767569878681001874e-9

/* [3/4] Gauss continued-fraction rational coeffs for tan(u) = u * P(u^2) / Q(u^2)
 * on |u| <= pi/4 (max abs err < 9e-16). */
#define CS_TAN_P0   1.0
#define CS_TAN_P1  -1.33333333333333331483e-1
#define CS_TAN_P2   3.41880341880341887353e-3
#define CS_TAN_P3  -1.77600177600177614536e-5

#define CS_TAN_Q0   1.0
#define CS_TAN_Q1  -4.66666666666666674068e-1
#define CS_TAN_Q2   2.56410256410256401692e-2
#define CS_TAN_Q3  -3.10800310800310798033e-4
#define CS_TAN_Q4   4.93333826667160027981e-7

/* =========================================================================
 * Scalar Estrin/Cephes exp and rational tan (exact parity reference)
 * ========================================================================= */

static inline double cs_scalar_pow2i(double k) {
    double val = k + CS_POW2_MAGIC;
    uint64_t u;
    memcpy(&u, &val, sizeof(u));
    u <<= 52;
    double out;
    memcpy(&out, &u, sizeof(out));
    return out;
}

static inline double cs_scalar_exp(double x) {
    if (x < -744.0) {
        return 0.0;
    }
    if (x > 709.7827) {
        return INFINITY;
    }
    double k = rint(x * CS_INV_LN2);
    double r = (x - k * CS_LN2_HI) - k * CS_LN2_LO;
    double r2 = r * r;
    double r4 = r2 * r2;
    double r8 = r4 * r4;

    double p01 = CS_EXP_C0 + CS_EXP_C1 * r;
    double p23 = CS_EXP_C2 + CS_EXP_C3 * r;
    double p45 = CS_EXP_C4 + CS_EXP_C5 * r;
    double p67 = CS_EXP_C6 + CS_EXP_C7 * r;
    double p89 = CS_EXP_C8 + CS_EXP_C9 * r;
    double p1011 = CS_EXP_C10 + CS_EXP_C11 * r;

    double p03 = p01 + r2 * p23;
    double p47 = p45 + r2 * p67;
    double p811 = p89 + r2 * p1011;
    double p07 = p03 + r4 * p47;
    double p812 = p811 + r4 * CS_EXP_C12;
    double poly = p07 + r8 * p812;

    double k1 = rint(0.5 * k);
    double k2 = k - k1;
    return (poly * cs_scalar_pow2i(k1)) * cs_scalar_pow2i(k2);
}

static inline double cs_scalar_tan_clamped(double theta) {
    double sign = (theta < 0.0) ? -1.0 : 1.0;
    double ax = fabs(theta);
    if (ax > CS_TAN_MAX_ARG) {
        ax = CS_TAN_MAX_ARG;
    }
    int flip = (ax > CS_PI_QUARTER);
    double u = flip ? ((CS_PI_HALF_HI - ax) + CS_PI_HALF_LO) : ax;
    double u2 = u * u;
    double p = CS_TAN_P0 + u2 * (CS_TAN_P1 + u2 * (CS_TAN_P2 + u2 * CS_TAN_P3));
    double q = CS_TAN_Q0 + u2 * (CS_TAN_Q1 + u2 * (CS_TAN_Q2 + u2 * (CS_TAN_Q3 + u2 * CS_TAN_Q4)));
    double up = u * p;
    double res = flip ? (q / up) : (up / q);
    return sign * res;
}

/* =========================================================================
 * ARM64 NEON SIMD Vector Primitives (float64x2_t)
 * ========================================================================= */

#if defined(CS_SIMD_NEON)

static inline float64x2_t cs_neon_pow2i(float64x2_t k) {
    float64x2_t val = vaddq_f64(k, vdupq_n_f64(CS_POW2_MAGIC));
    uint64x2_t u = vreinterpretq_u64_f64(val);
    u = vshlq_n_u64(u, 52);
    return vreinterpretq_f64_u64(u);
}

static inline float64x2_t cs_vexp_f64(float64x2_t x) {
    const float64x2_t lo_lim = vdupq_n_f64(-744.0);
    const float64x2_t hi_lim = vdupq_n_f64(709.7827);
    uint64x2_t under = vcltq_f64(x, lo_lim);
    uint64x2_t over  = vcgtq_f64(x, hi_lim);

    float64x2_t xc = vmaxq_f64(vminq_f64(x, hi_lim), lo_lim);
    float64x2_t k = vrndnq_f64(vmulq_f64(xc, vdupq_n_f64(CS_INV_LN2)));
    float64x2_t r = vfmsq_f64(xc, k, vdupq_n_f64(CS_LN2_HI));
    r = vfmsq_f64(r, k, vdupq_n_f64(CS_LN2_LO));

    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t r4 = vmulq_f64(r2, r2);
    float64x2_t r8 = vmulq_f64(r4, r4);

    float64x2_t p01 = vfmaq_f64(vdupq_n_f64(CS_EXP_C0), vdupq_n_f64(CS_EXP_C1), r);
    float64x2_t p23 = vfmaq_f64(vdupq_n_f64(CS_EXP_C2), vdupq_n_f64(CS_EXP_C3), r);
    float64x2_t p45 = vfmaq_f64(vdupq_n_f64(CS_EXP_C4), vdupq_n_f64(CS_EXP_C5), r);
    float64x2_t p67 = vfmaq_f64(vdupq_n_f64(CS_EXP_C6), vdupq_n_f64(CS_EXP_C7), r);
    float64x2_t p89 = vfmaq_f64(vdupq_n_f64(CS_EXP_C8), vdupq_n_f64(CS_EXP_C9), r);
    float64x2_t p1011 = vfmaq_f64(vdupq_n_f64(CS_EXP_C10), vdupq_n_f64(CS_EXP_C11), r);

    float64x2_t p03  = vfmaq_f64(p01, r2, p23);
    float64x2_t p47  = vfmaq_f64(p45, r2, p67);
    float64x2_t p811 = vfmaq_f64(p89, r2, p1011);
    float64x2_t p07  = vfmaq_f64(p03, r4, p47);
    float64x2_t p812 = vfmaq_f64(p811, r4, vdupq_n_f64(CS_EXP_C12));
    float64x2_t poly = vfmaq_f64(p07, r8, p812);

    float64x2_t k1 = vrndnq_f64(vmulq_f64(vdupq_n_f64(0.5), k));
    float64x2_t k2 = vsubq_f64(k, k1);
    float64x2_t res = vmulq_f64(vmulq_f64(poly, cs_neon_pow2i(k1)), cs_neon_pow2i(k2));

    res = vbslq_f64(under, vdupq_n_f64(0.0), res);
    res = vbslq_f64(over, vdupq_n_f64(INFINITY), res);
    return res;
}

static inline float64x2_t cs_vtan_clamped_f64(float64x2_t theta) {
    const uint64x2_t sign_mask = vdupq_n_u64(0x8000000000000000ULL);
    uint64x2_t sign_bits = vandq_u64(vreinterpretq_u64_f64(theta), sign_mask);
    float64x2_t ax = vabsq_f64(theta);
    ax = vminq_f64(ax, vdupq_n_f64(CS_TAN_MAX_ARG));

    uint64x2_t flip = vcgtq_f64(ax, vdupq_n_f64(CS_PI_QUARTER));
    float64x2_t u_flip = vaddq_f64(vsubq_f64(vdupq_n_f64(CS_PI_HALF_HI), ax),
                                   vdupq_n_f64(CS_PI_HALF_LO));
    float64x2_t u = vbslq_f64(flip, u_flip, ax);
    float64x2_t u2 = vmulq_f64(u, u);
    float64x2_t u4 = vmulq_f64(u2, u2);
    float64x2_t u8 = vmulq_f64(u4, u4);

    /* Estrin evaluation cuts FMA latency depth vs Horner while keeping < 1e-15 accuracy */
    float64x2_t p01 = vfmaq_f64(vdupq_n_f64(CS_TAN_P0), u2, vdupq_n_f64(CS_TAN_P1));
    float64x2_t p23 = vfmaq_f64(vdupq_n_f64(CS_TAN_P2), u2, vdupq_n_f64(CS_TAN_P3));
    float64x2_t p = vfmaq_f64(p01, u4, p23);

    float64x2_t q01 = vfmaq_f64(vdupq_n_f64(CS_TAN_Q0), u2, vdupq_n_f64(CS_TAN_Q1));
    float64x2_t q23 = vfmaq_f64(vdupq_n_f64(CS_TAN_Q2), u2, vdupq_n_f64(CS_TAN_Q3));
    float64x2_t q03 = vfmaq_f64(q01, u4, q23);
    float64x2_t q = vfmaq_f64(q03, u8, vdupq_n_f64(CS_TAN_Q4));

    float64x2_t up = vmulq_f64(u, p);
    float64x2_t num = vbslq_f64(flip, q, up);
    float64x2_t den = vbslq_f64(flip, up, q);
    float64x2_t mag = vdivq_f64(num, den);

    return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(mag), sign_bits));
}

#elif defined(CS_SIMD_AVX2)

static inline __m256d cs_avx_fmadd(__m256d a, __m256d b, __m256d c) {
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    return _mm256_fmadd_pd(a, b, c);
#else
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
}

static inline __m256d cs_avx_fnmadd(__m256d a, __m256d b, __m256d c) {
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    return _mm256_fnmadd_pd(a, b, c);
#else
    return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#endif
}

static inline __m256d cs_avx_pow2i(__m256d k) {
    __m256d val = _mm256_add_pd(k, _mm256_set1_pd(CS_POW2_MAGIC));
    __m256i u = _mm256_castpd_si256(val);
    u = _mm256_slli_epi64(u, 52);
    return _mm256_castsi256_pd(u);
}

static inline __m256d cs_vexp_f64(__m256d x) {
    const __m256d lo_lim = _mm256_set1_pd(-744.0);
    const __m256d hi_lim = _mm256_set1_pd(709.7827);
    __m256d under = _mm256_cmp_pd(x, lo_lim, _CMP_LT_OQ);
    __m256d over  = _mm256_cmp_pd(x, hi_lim, _CMP_GT_OQ);

    __m256d xc = _mm256_max_pd(_mm256_min_pd(x, hi_lim), lo_lim);
    __m256d k = _mm256_round_pd(
        _mm256_mul_pd(xc, _mm256_set1_pd(CS_INV_LN2)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = cs_avx_fnmadd(k, _mm256_set1_pd(CS_LN2_HI), xc);
    r = cs_avx_fnmadd(k, _mm256_set1_pd(CS_LN2_LO), r);

    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d r4 = _mm256_mul_pd(r2, r2);
    __m256d r8 = _mm256_mul_pd(r4, r4);

    __m256d p01   = cs_avx_fmadd(_mm256_set1_pd(CS_EXP_C1), r, _mm256_set1_pd(CS_EXP_C0));
    __m256d p23   = cs_avx_fmadd(_mm256_set1_pd(CS_EXP_C3), r, _mm256_set1_pd(CS_EXP_C2));
    __m256d p45   = cs_avx_fmadd(_mm256_set1_pd(CS_EXP_C5), r, _mm256_set1_pd(CS_EXP_C4));
    __m256d p67   = cs_avx_fmadd(_mm256_set1_pd(CS_EXP_C7), r, _mm256_set1_pd(CS_EXP_C6));
    __m256d p89   = cs_avx_fmadd(_mm256_set1_pd(CS_EXP_C9), r, _mm256_set1_pd(CS_EXP_C8));
    __m256d p1011 = cs_avx_fmadd(_mm256_set1_pd(CS_EXP_C11), r, _mm256_set1_pd(CS_EXP_C10));

    __m256d p03  = cs_avx_fmadd(r2, p23, p01);
    __m256d p47  = cs_avx_fmadd(r2, p67, p45);
    __m256d p811 = cs_avx_fmadd(r2, p1011, p89);
    __m256d p07  = cs_avx_fmadd(r4, p47, p03);
    __m256d p812 = cs_avx_fmadd(r4, _mm256_set1_pd(CS_EXP_C12), p811);
    __m256d poly = cs_avx_fmadd(r8, p812, p07);

    __m256d k1 = _mm256_round_pd(
        _mm256_mul_pd(_mm256_set1_pd(0.5), k),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d k2 = _mm256_sub_pd(k, k1);
    __m256d res = _mm256_mul_pd(_mm256_mul_pd(poly, cs_avx_pow2i(k1)), cs_avx_pow2i(k2));

    res = _mm256_blendv_pd(res, _mm256_setzero_pd(), under);
    res = _mm256_blendv_pd(res, _mm256_set1_pd(INFINITY), over);
    return res;
}

static inline __m256d cs_vtan_clamped_f64(__m256d theta) {
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d sign_bits = _mm256_and_pd(theta, sign_mask);
    __m256d ax = _mm256_andnot_pd(sign_mask, theta);
    ax = _mm256_min_pd(ax, _mm256_set1_pd(CS_TAN_MAX_ARG));

    __m256d flip = _mm256_cmp_pd(ax, _mm256_set1_pd(CS_PI_QUARTER), _CMP_GT_OQ);
    __m256d u_flip = _mm256_add_pd(
        _mm256_sub_pd(_mm256_set1_pd(CS_PI_HALF_HI), ax),
        _mm256_set1_pd(CS_PI_HALF_LO));
    __m256d u = _mm256_blendv_pd(ax, u_flip, flip);
    __m256d u2 = _mm256_mul_pd(u, u);
    __m256d u4 = _mm256_mul_pd(u2, u2);
    __m256d u8 = _mm256_mul_pd(u4, u4);

    __m256d p01 = cs_avx_fmadd(u2, _mm256_set1_pd(CS_TAN_P1), _mm256_set1_pd(CS_TAN_P0));
    __m256d p23 = cs_avx_fmadd(u2, _mm256_set1_pd(CS_TAN_P3), _mm256_set1_pd(CS_TAN_P2));
    __m256d p = cs_avx_fmadd(u4, p23, p01);

    __m256d q01 = cs_avx_fmadd(u2, _mm256_set1_pd(CS_TAN_Q1), _mm256_set1_pd(CS_TAN_Q0));
    __m256d q23 = cs_avx_fmadd(u2, _mm256_set1_pd(CS_TAN_Q3), _mm256_set1_pd(CS_TAN_Q2));
    __m256d q03 = cs_avx_fmadd(u4, q23, q01);
    __m256d q = cs_avx_fmadd(u8, _mm256_set1_pd(CS_TAN_Q4), q03);

    __m256d up = _mm256_mul_pd(u, p);
    __m256d num = _mm256_blendv_pd(up, q, flip);
    __m256d den = _mm256_blendv_pd(q, up, flip);
    __m256d mag = _mm256_div_pd(num, den);

    return _mm256_or_pd(mag, sign_bits);
}

#endif /* SIMD backend */

#endif /* CS_SIMD_H */
