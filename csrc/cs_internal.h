#ifndef CS_INTERNAL_H
#define CS_INTERNAL_H

#include "chronosieve.h"
#include "cs_simd.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
  #define CS_RESTRICT __restrict
#else
  #define CS_RESTRICT restrict
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define CS_LIKELY(x)   __builtin_expect(!!(x), 1)
  #define CS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define CS_LIKELY(x)   (x)
  #define CS_UNLIKELY(x) (x)
#endif

/* Internal fast-path helper (implemented in edetector.c) for per-agent Catoni rogue updates. */
cs_status_t cs_edet_step_catoni_esr_fast(
    cs_edetector_t *det,
    double x,
    double *out_stat,
    bool *out_alarmed);

/* =========================================================================
 * Portable 64-byte Aligned Allocation (C11/macOS/Linux/ASan safe)
 * ========================================================================= */

static inline void *cs_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }
    if (size == 0) {
        size = alignment;
    }
    size_t total = size + alignment + sizeof(void *);
    if (total < size) {
        return NULL;
    }
    void *raw = malloc(total);
    if (!raw) {
        return NULL;
    }
    uintptr_t addr = (uintptr_t)raw + sizeof(void *);
    uintptr_t aligned = (addr + (alignment - 1U)) & ~(uintptr_t)(alignment - 1U);
    ((void **)aligned)[-1] = raw;
    return (void *)aligned;
}

static inline void *cs_aligned_calloc(size_t alignment, size_t count, size_t elem_size) {
    if (count > 0 && elem_size > SIZE_MAX / count) {
        return NULL;
    }
    size_t total = count * elem_size;
    void *ptr = cs_aligned_alloc(alignment, total);
    if (ptr && total > 0) {
        memset(ptr, 0, total);
    }
    return ptr;
}

static inline void cs_aligned_free(void *ptr) {
    if (ptr) {
        void *raw = ((void **)ptr)[-1];
        free(raw);
    }
}

/* =========================================================================
 * Deterministic PRNG: splitmix64 + xoshiro256**
 * ========================================================================= */

static inline uint64_t cs_splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline double cs_u64_to_uniform01(uint64_t x) {
    return (double)(x >> 11) * 0x1.0p-53;
}

typedef struct {
    uint64_t s[4];
} cs_rng_t;

static inline uint64_t cs_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline void cs_rng_seed(cs_rng_t *rng, uint64_t seed) {
    uint64_t sm = (seed == 0) ? 0x6a09e667f3bcc909ULL : seed;
    rng->s[0] = cs_splitmix64_next(&sm);
    rng->s[1] = cs_splitmix64_next(&sm);
    rng->s[2] = cs_splitmix64_next(&sm);
    rng->s[3] = cs_splitmix64_next(&sm);
}

static inline uint64_t cs_rng_u64(cs_rng_t *rng) {
    const uint64_t result = cs_rotl64(rng->s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = cs_rotl64(rng->s[3], 45);
    return result;
}

static inline double cs_rng_uniform01(cs_rng_t *rng) {
    return cs_u64_to_uniform01(cs_rng_u64(rng));
}

/* =========================================================================
 * Cache-Line-Friendly Branchless SIMD O(log n) Order-Statistic B-Tree
 * - CS_BTREE_T = 8 -> 15 max keys, 16 padded key slots (128B = 1 M4 cache line)
 * - Inline child_sizes[] eliminates sibling cache-line misses during rank descent
 * - Single-pass cs_btree_rank_and_insert() fuses query + insert into 1 traversal
 * - Supports exact duplicate multiplicities and O(1) deterministic epoch reset
 * ========================================================================= */

#define CS_BTREE_T 8U
#define CS_BTREE_MAX_KEYS (2U * CS_BTREE_T - 1U)      /* 15 */
#define CS_BTREE_KEY_SLOTS (2U * CS_BTREE_T)          /* 16 (SIMD aligned, unused = +INF) */
#define CS_BTREE_MAX_CHILDREN (2U * CS_BTREE_T)       /* 16 */

typedef struct {
    double keys[CS_BTREE_KEY_SLOTS];              /* 128B: cache line 0 (leaf + internal) */
    uint32_t counts[CS_BTREE_KEY_SLOTS];          /*  64B: cache line 1 lo (leaf + internal) */
    uint32_t subtree_size;                        /*   4B */
    uint16_t num_keys;                            /*   2B */
    uint8_t is_leaf;                              /*   1B */
    uint8_t has_dups;                             /*   1B */
    uint32_t pad[14];                             /*  56B -> cache line 1 ends at 256B */
    uint32_t child_sizes[CS_BTREE_MAX_CHILDREN];  /*  64B: cache line 2 lo (internal only!) */
    int32_t children[CS_BTREE_MAX_CHILDREN];      /*  64B: cache line 2 hi (internal only!) */
} cs_btree_node_t;

#if defined(CS_SIMD_NEON) || defined(CS_SIMD_AVX2)
static const uint32_t CS_BTREE_MASK16[16][16] = {
    {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U},
    {~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, 0U}
};
#endif

static inline uint32_t cs_btree_sum_internal_lt(const cs_btree_node_t *n, uint16_t i) {
#if defined(CS_SIMD_NEON)
    const uint32_t *mk = CS_BTREE_MASK16[i];
    uint32x4_t s0 = vandq_u32(
        vaddq_u32(vld1q_u32(n->counts + 0), vld1q_u32(n->child_sizes + 0)), vld1q_u32(mk + 0));
    uint32x4_t s1 = vandq_u32(
        vaddq_u32(vld1q_u32(n->counts + 4), vld1q_u32(n->child_sizes + 4)), vld1q_u32(mk + 4));
    uint32x4_t s2 = vandq_u32(
        vaddq_u32(vld1q_u32(n->counts + 8), vld1q_u32(n->child_sizes + 8)), vld1q_u32(mk + 8));
    uint32x4_t s3 = vandq_u32(
        vaddq_u32(vld1q_u32(n->counts + 12), vld1q_u32(n->child_sizes + 12)), vld1q_u32(mk + 12));
    return vaddvq_u32(vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3)));
#elif defined(CS_SIMD_AVX2)
    const uint32_t *mk = CS_BTREE_MASK16[i];
    __m256i s0 = _mm256_and_si256(
        _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i *)(n->counts + 0)),
            _mm256_loadu_si256((const __m256i *)(n->child_sizes + 0))),
        _mm256_loadu_si256((const __m256i *)(mk + 0)));
    __m256i s1 = _mm256_and_si256(
        _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i *)(n->counts + 8)),
            _mm256_loadu_si256((const __m256i *)(n->child_sizes + 8))),
        _mm256_loadu_si256((const __m256i *)(mk + 8)));
    __m256i s = _mm256_add_epi32(s0, s1);
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, s);
    return (tmp[0] + tmp[1] + tmp[2] + tmp[3]) + (tmp[4] + tmp[5] + tmp[6] + tmp[7]);
#else
    uint32_t acc = 0U;
    for (uint16_t j = 0; j < i; ++j) {
        acc += n->child_sizes[j] + n->counts[j];
    }
    return acc;
#endif
}

static inline uint32_t cs_btree_sum_counts_lt(const cs_btree_node_t *n, uint16_t i) {
#if defined(CS_SIMD_NEON)
    const uint32_t *mk = CS_BTREE_MASK16[i];
    uint32x4_t s0 = vandq_u32(vld1q_u32(n->counts + 0), vld1q_u32(mk + 0));
    uint32x4_t s1 = vandq_u32(vld1q_u32(n->counts + 4), vld1q_u32(mk + 4));
    uint32x4_t s2 = vandq_u32(vld1q_u32(n->counts + 8), vld1q_u32(mk + 8));
    uint32x4_t s3 = vandq_u32(vld1q_u32(n->counts + 12), vld1q_u32(mk + 12));
    return vaddvq_u32(vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3)));
#elif defined(CS_SIMD_AVX2)
    const uint32_t *mk = CS_BTREE_MASK16[i];
    __m256i s0 = _mm256_and_si256(
        _mm256_loadu_si256((const __m256i *)(n->counts + 0)),
        _mm256_loadu_si256((const __m256i *)(mk + 0)));
    __m256i s1 = _mm256_and_si256(
        _mm256_loadu_si256((const __m256i *)(n->counts + 8)),
        _mm256_loadu_si256((const __m256i *)(mk + 8)));
    __m256i s = _mm256_add_epi32(s0, s1);
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, s);
    return (tmp[0] + tmp[1] + tmp[2] + tmp[3]) + (tmp[4] + tmp[5] + tmp[6] + tmp[7]);
#else
    uint32_t acc = 0U;
    for (uint16_t j = 0; j < i; ++j) {
        acc += n->counts[j];
    }
    return acc;
#endif
}

typedef struct {
    cs_btree_node_t *nodes;
    size_t pool_used;
    size_t pool_cap;
    int32_t root;
    uint32_t total_count;
} cs_btree_t;

static inline void cs_btree_init(cs_btree_t *bt) {
    bt->nodes = NULL;
    bt->pool_used = 0;
    bt->pool_cap = 0;
    bt->root = -1;
    bt->total_count = 0;
}

static inline void cs_btree_free(cs_btree_t *bt) {
    cs_aligned_free(bt->nodes);
    bt->nodes = NULL;
    bt->pool_used = 0;
    bt->pool_cap = 0;
    bt->root = -1;
    bt->total_count = 0;
}

static inline void cs_btree_clear(cs_btree_t *bt) {
    bt->pool_used = 0;
    bt->root = -1;
    bt->total_count = 0;
}

static inline cs_status_t cs_btree_reserve(cs_btree_t *bt, size_t extra) {
    if (CS_LIKELY(bt->pool_used + extra <= bt->pool_cap)) {
        return CS_OK;
    }
    size_t new_cap = (bt->pool_cap == 0) ? 32U : bt->pool_cap * 2U;
    while (new_cap < bt->pool_used + extra) {
        if (new_cap > SIZE_MAX / 2U) {
            return CS_ERR_NOMEM;
        }
        new_cap *= 2U;
    }
    if (new_cap > SIZE_MAX / sizeof(cs_btree_node_t)) {
        return CS_ERR_NOMEM;
    }
    cs_btree_node_t *new_nodes = (cs_btree_node_t *)cs_aligned_alloc(
        128U, new_cap * sizeof(cs_btree_node_t));
    if (!new_nodes) {
        return CS_ERR_NOMEM;
    }
    if (bt->nodes && bt->pool_used > 0U) {
        memcpy(new_nodes, bt->nodes, bt->pool_used * sizeof(cs_btree_node_t));
    }
    cs_aligned_free(bt->nodes);
    bt->nodes = new_nodes;
    bt->pool_cap = new_cap;
    return CS_OK;
}

static inline int32_t cs_btree_alloc_node_unchecked(cs_btree_t *bt) {
    int32_t idx = (int32_t)bt->pool_used++;
    cs_btree_node_t *n = &bt->nodes[idx];
    for (uint16_t k = 0; k < CS_BTREE_KEY_SLOTS; ++k) {
        n->keys[k] = INFINITY;
        n->counts[k] = 0U;
        n->child_sizes[k] = 0U;
    }
    n->subtree_size = 0U;
    n->num_keys = 0U;
    n->is_leaf = 1U;
    n->has_dups = 0U;
    return idx;
}

/* Branchless SIMD search: returns exact count of keys strictly < x in [0, num_keys].
 * Relies on invariant keys[num_keys .. 15] == +INFINITY (never < finite x). */
static inline uint16_t cs_btree_find_pos(const double *keys, uint16_t num_keys, double x) {
#if defined(CS_SIMD_NEON)
    (void)num_keys;
    float64x2_t vx = vdupq_n_f64(x);
    int64x2_t c0 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 0), vx));
    int64x2_t c1 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 2), vx));
    int64x2_t c2 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 4), vx));
    int64x2_t c3 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 6), vx));
    int64x2_t c4 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 8), vx));
    int64x2_t c5 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 10), vx));
    int64x2_t c6 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 12), vx));
    int64x2_t c7 = vreinterpretq_s64_u64(vcltq_f64(vld1q_f64(keys + 14), vx));
    int64x2_t s01 = vaddq_s64(c0, c1);
    int64x2_t s23 = vaddq_s64(c2, c3);
    int64x2_t s45 = vaddq_s64(c4, c5);
    int64x2_t s67 = vaddq_s64(c6, c7);
    int64x2_t s03 = vaddq_s64(s01, s23);
    int64x2_t s47 = vaddq_s64(s45, s67);
    return (uint16_t)(-(int)vaddvq_s64(vaddq_s64(s03, s47)));
#elif defined(CS_SIMD_AVX2)
    static const uint8_t pop4[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    (void)num_keys;
    __m256d vx = _mm256_set1_pd(x);
    int m0 = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(keys + 0), vx, _CMP_LT_OQ));
    int m1 = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(keys + 4), vx, _CMP_LT_OQ));
    int m2 = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(keys + 8), vx, _CMP_LT_OQ));
    int m3 = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(keys + 12), vx, _CMP_LT_OQ));
    return (uint16_t)(pop4[m0] + pop4[m1] + pop4[m2] + pop4[m3]);
#else
    uint16_t lo = 0U;
    uint16_t hi = num_keys;
    while (lo < hi) {
        uint16_t mid = (uint16_t)((lo + hi) >> 1);
        if (keys[mid] < x) {
            lo = (uint16_t)(mid + 1U);
        } else {
            hi = mid;
        }
    }
    return lo;
#endif
}

static inline void cs_btree_split_child(cs_btree_t *bt, int32_t parent_idx, uint16_t child_pos) {
    int32_t right_idx = cs_btree_alloc_node_unchecked(bt);
    cs_btree_node_t *parent = &bt->nodes[parent_idx];
    int32_t left_idx = parent->children[child_pos];
    cs_btree_node_t *left = &bt->nodes[left_idx];
    cs_btree_node_t *right = &bt->nodes[right_idx];

    right->is_leaf = left->is_leaf;
    right->num_keys = (uint16_t)(CS_BTREE_T - 1U);

    uint32_t left_sz = 0U;
    uint32_t right_sz = 0U;
    uint8_t left_dups = 0U;
    uint8_t right_dups = 0U;

    for (uint16_t j = 0; j < CS_BTREE_T - 1U; ++j) {
        uint32_t lc = left->counts[j];
        left_sz += lc;
        if (lc > 1U) {
            left_dups = 1U;
        }
        double rk = left->keys[j + CS_BTREE_T];
        uint32_t rc = left->counts[j + CS_BTREE_T];
        right->keys[j] = rk;
        right->counts[j] = rc;
        right_sz += rc;
        if (rc > 1U) {
            right_dups = 1U;
        }
    }

    double med_key = left->keys[CS_BTREE_T - 1U];
    uint32_t med_cnt = left->counts[CS_BTREE_T - 1U];

    for (uint16_t j = CS_BTREE_T - 1U; j < CS_BTREE_MAX_KEYS; ++j) {
        left->keys[j] = INFINITY;
    }

    if (!left->is_leaf) {
        for (uint16_t j = 0; j < CS_BTREE_T; ++j) {
            left_sz += left->child_sizes[j];
            right->children[j] = left->children[j + CS_BTREE_T];
            uint32_t rcs = left->child_sizes[j + CS_BTREE_T];
            right->child_sizes[j] = rcs;
            right_sz += rcs;
        }
    }

    left->num_keys = (uint16_t)(CS_BTREE_T - 1U);
    left->subtree_size = left_sz;
    left->has_dups = left_dups;
    right->subtree_size = right_sz;
    right->has_dups = right_dups;

    for (uint16_t j = parent->num_keys; j > child_pos; --j) {
        parent->children[j + 1U] = parent->children[j];
        parent->child_sizes[j + 1U] = parent->child_sizes[j];
    }
    parent->children[child_pos + 1U] = right_idx;
    parent->child_sizes[child_pos] = left_sz;
    parent->child_sizes[child_pos + 1U] = right_sz;

    for (uint16_t j = parent->num_keys; j > child_pos; --j) {
        parent->keys[j] = parent->keys[j - 1U];
        parent->counts[j] = parent->counts[j - 1U];
    }
    parent->keys[child_pos] = med_key;
    parent->counts[child_pos] = med_cnt;
    if (med_cnt > 1U) {
        parent->has_dups = 1U;
    }
    parent->num_keys++;
}

static inline void cs_btree_rank_query(
    const cs_btree_t *bt,
    double x,
    uint32_t *out_less,
    uint32_t *out_equal)
{
    uint32_t less = 0U;
    uint32_t equal = 0U;
    int32_t cur = bt->root;
    while (cur >= 0) {
        const cs_btree_node_t *n = &bt->nodes[cur];
        uint16_t i = cs_btree_find_pos(n->keys, n->num_keys, x);
        if (n->is_leaf) {
            less += n->has_dups ? cs_btree_sum_counts_lt(n, i) : (uint32_t)i;
            if (i < n->num_keys && n->keys[i] == x) {
                equal = n->counts[i];
            }
            break;
        }
        less += cs_btree_sum_internal_lt(n, i);
        if (i < n->num_keys && n->keys[i] == x) {
            less += n->child_sizes[i];
            equal = n->counts[i];
            break;
        }
        cur = n->children[i];
    }
    *out_less = less;
    *out_equal = equal;
}

/* Single-pass fused rank query (pre-insertion) + insert (1 top-down traversal!). */
static inline cs_status_t cs_btree_rank_and_insert(
    cs_btree_t *bt,
    double x,
    uint32_t *out_less,
    uint32_t *out_equal)
{
    /* Max height for 2^32 keys with T=8 is < 12; reserving 16 nodes avoids mid-loop realloc. */
    cs_status_t st = cs_btree_reserve(bt, 16U);
    if (CS_UNLIKELY(st != CS_OK)) {
        return st;
    }

    if (CS_UNLIKELY(bt->root < 0)) {
        int32_t r = cs_btree_alloc_node_unchecked(bt);
        cs_btree_node_t *rn = &bt->nodes[r];
        rn->keys[0] = x;
        rn->counts[0] = 1U;
        rn->num_keys = 1U;
        rn->is_leaf = 1U;
        rn->has_dups = 0U;
        rn->subtree_size = 1U;
        bt->root = r;
        bt->total_count = 1U;
        *out_less = 0U;
        *out_equal = 0U;
        return CS_OK;
    }

    if (CS_UNLIKELY(bt->nodes[bt->root].num_keys == CS_BTREE_MAX_KEYS)) {
        int32_t old_root = bt->root;
        int32_t new_root = cs_btree_alloc_node_unchecked(bt);
        cs_btree_node_t *nr = &bt->nodes[new_root];
        uint32_t old_sz = bt->nodes[old_root].subtree_size;
        nr->is_leaf = 0U;
        nr->num_keys = 0U;
        nr->children[0] = old_root;
        nr->child_sizes[0] = old_sz;
        nr->subtree_size = old_sz;
        bt->root = new_root;
        cs_btree_split_child(bt, new_root, 0U);
    }

    uint32_t less = 0U;
    uint32_t equal = 0U;
    int32_t cur = bt->root;
    for (;;) {
        cs_btree_node_t *n = &bt->nodes[cur];
        uint16_t i = cs_btree_find_pos(n->keys, n->num_keys, x);

        if (n->is_leaf) {
            less += CS_UNLIKELY(n->has_dups) ? cs_btree_sum_counts_lt(n, i) : (uint32_t)i;
            n->subtree_size++;
            if (CS_UNLIKELY(i < n->num_keys && n->keys[i] == x)) {
                equal = n->counts[i];
                n->counts[i]++;
                n->has_dups = 1U;
                break;
            }
            size_t tail = (size_t)(n->num_keys - i);
            if (tail > 0U) {
                memmove(&n->keys[i + 1U], &n->keys[i], tail * sizeof(double));
                if (CS_UNLIKELY(n->has_dups)) {
                    memmove(&n->counts[i + 1U], &n->counts[i], tail * sizeof(uint32_t));
                    n->counts[i] = 1U;
                } else {
                    n->counts[n->num_keys] = 1U;
                }
            } else {
                n->counts[i] = 1U;
            }
            n->keys[i] = x;
            n->num_keys++;
            break;
        }

        less += cs_btree_sum_internal_lt(n, i);
        if (CS_UNLIKELY(i < n->num_keys && n->keys[i] == x)) {
            less += n->child_sizes[i];
            equal = n->counts[i];
            n->counts[i]++;
            n->has_dups = 1U;
            n->subtree_size++;
            break;
        }

        int32_t child_idx = n->children[i];
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&bt->nodes[child_idx], 0, 3);
#endif
        if (CS_UNLIKELY(bt->nodes[child_idx].num_keys == CS_BTREE_MAX_KEYS)) {
            cs_btree_split_child(bt, cur, i);
            if (x == n->keys[i]) {
                less += n->child_sizes[i];
                equal = n->counts[i];
                n->counts[i]++;
                n->has_dups = 1U;
                n->subtree_size++;
                break;
            }
            if (x > n->keys[i]) {
                less += n->child_sizes[i] + n->counts[i];
                ++i;
            }
            child_idx = n->children[i];
        }
        n->subtree_size++;
        n->child_sizes[i]++;
        cur = child_idx;
    }

    bt->total_count++;
    *out_less = less;
    *out_equal = equal;
    return CS_OK;
}

static inline cs_status_t cs_btree_insert(cs_btree_t *bt, double x) {
    uint32_t less = 0U, equal = 0U;
    return cs_btree_rank_and_insert(bt, x, &less, &equal);
}

#endif /* CS_INTERNAL_H */
