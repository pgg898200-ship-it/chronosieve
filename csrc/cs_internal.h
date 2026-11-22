#ifndef CS_INTERNAL_H
#define CS_INTERNAL_H

#include "chronosieve.h"
#include "cs_simd.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
 * Cache-Friendly O(log n) Order-Statistic B-Tree (Arena-Backed)
 * Supports exact duplicate multiplicities and O(1) deterministic epoch reset.
 * ========================================================================= */

#define CS_BTREE_T 16U
#define CS_BTREE_MAX_KEYS (2U * CS_BTREE_T - 1U)      /* 31 */
#define CS_BTREE_MAX_CHILDREN (2U * CS_BTREE_T)       /* 32 */

typedef struct {
    double keys[CS_BTREE_MAX_KEYS];
    uint32_t counts[CS_BTREE_MAX_KEYS];
    int32_t children[CS_BTREE_MAX_CHILDREN];
    uint32_t subtree_size;
    uint16_t num_keys;
    uint8_t is_leaf;
    uint8_t pad;
} cs_btree_node_t;

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
    free(bt->nodes);
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
    if (bt->pool_used + extra <= bt->pool_cap) {
        return CS_OK;
    }
    size_t new_cap = (bt->pool_cap == 0) ? 16U : bt->pool_cap * 2U;
    while (new_cap < bt->pool_used + extra) {
        if (new_cap > SIZE_MAX / 2U) {
            return CS_ERR_NOMEM;
        }
        new_cap *= 2U;
    }
    if (new_cap > SIZE_MAX / sizeof(cs_btree_node_t)) {
        return CS_ERR_NOMEM;
    }
    cs_btree_node_t *new_nodes = (cs_btree_node_t *)realloc(
        bt->nodes, new_cap * sizeof(cs_btree_node_t));
    if (!new_nodes) {
        return CS_ERR_NOMEM;
    }
    bt->nodes = new_nodes;
    bt->pool_cap = new_cap;
    return CS_OK;
}

static inline int32_t cs_btree_alloc_node_unchecked(cs_btree_t *bt) {
    int32_t idx = (int32_t)bt->pool_used++;
    cs_btree_node_t *n = &bt->nodes[idx];
    n->subtree_size = 0;
    n->num_keys = 0;
    n->is_leaf = 1;
    n->pad = 0;
    return idx;
}

static inline void cs_btree_recompute_size(cs_btree_t *bt, int32_t node_idx) {
    cs_btree_node_t *n = &bt->nodes[node_idx];
    uint32_t sz = 0;
    for (uint16_t i = 0; i < n->num_keys; ++i) {
        sz += n->counts[i];
    }
    if (!n->is_leaf) {
        for (uint16_t i = 0; i <= n->num_keys; ++i) {
            sz += bt->nodes[n->children[i]].subtree_size;
        }
    }
    n->subtree_size = sz;
}

static inline void cs_btree_split_child(cs_btree_t *bt, int32_t parent_idx, uint16_t child_pos) {
    int32_t right_idx = cs_btree_alloc_node_unchecked(bt);
    cs_btree_node_t *parent = &bt->nodes[parent_idx];
    int32_t left_idx = parent->children[child_pos];
    cs_btree_node_t *left = &bt->nodes[left_idx];
    cs_btree_node_t *right = &bt->nodes[right_idx];

    right->is_leaf = left->is_leaf;
    right->num_keys = (uint16_t)(CS_BTREE_T - 1U);

    for (uint16_t j = 0; j < CS_BTREE_T - 1U; ++j) {
        right->keys[j] = left->keys[j + CS_BTREE_T];
        right->counts[j] = left->counts[j + CS_BTREE_T];
    }
    if (!left->is_leaf) {
        for (uint16_t j = 0; j < CS_BTREE_T; ++j) {
            right->children[j] = left->children[j + CS_BTREE_T];
        }
    }

    double med_key = left->keys[CS_BTREE_T - 1U];
    uint32_t med_cnt = left->counts[CS_BTREE_T - 1U];
    left->num_keys = (uint16_t)(CS_BTREE_T - 1U);

    for (uint16_t j = parent->num_keys; j > child_pos; --j) {
        parent->children[j + 1U] = parent->children[j];
    }
    parent->children[child_pos + 1U] = right_idx;

    for (uint16_t j = parent->num_keys; j > child_pos; --j) {
        parent->keys[j] = parent->keys[j - 1U];
        parent->counts[j] = parent->counts[j - 1U];
    }
    parent->keys[child_pos] = med_key;
    parent->counts[child_pos] = med_cnt;
    parent->num_keys++;

    cs_btree_recompute_size(bt, left_idx);
    cs_btree_recompute_size(bt, right_idx);
}

static inline void cs_btree_rank_query(
    const cs_btree_t *bt,
    double x,
    uint32_t *out_less,
    uint32_t *out_equal)
{
    uint32_t less = 0;
    uint32_t equal = 0;
    int32_t cur = bt->root;
    while (cur >= 0) {
        const cs_btree_node_t *n = &bt->nodes[cur];
        uint16_t i = 0;
        while (i < n->num_keys && n->keys[i] < x) {
            if (!n->is_leaf) {
                less += bt->nodes[n->children[i]].subtree_size;
            }
            less += n->counts[i];
            ++i;
        }
        if (i < n->num_keys && n->keys[i] == x) {
            if (!n->is_leaf) {
                less += bt->nodes[n->children[i]].subtree_size;
            }
            equal = n->counts[i];
            break;
        }
        if (n->is_leaf) {
            break;
        }
        cur = n->children[i];
    }
    *out_less = less;
    *out_equal = equal;
}

static inline cs_status_t cs_btree_insert(cs_btree_t *bt, double x) {
    /* Max height for 2^32 keys with T=16 is < 9; reserving 12 nodes guarantees no realloc in loop. */
    cs_status_t st = cs_btree_reserve(bt, 12U);
    if (st != CS_OK) {
        return st;
    }

    if (bt->root < 0) {
        int32_t r = cs_btree_alloc_node_unchecked(bt);
        cs_btree_node_t *rn = &bt->nodes[r];
        rn->keys[0] = x;
        rn->counts[0] = 1U;
        rn->num_keys = 1U;
        rn->is_leaf = 1U;
        rn->subtree_size = 1U;
        bt->root = r;
        bt->total_count = 1U;
        return CS_OK;
    }

    if (bt->nodes[bt->root].num_keys == CS_BTREE_MAX_KEYS) {
        int32_t old_root = bt->root;
        int32_t new_root = cs_btree_alloc_node_unchecked(bt);
        cs_btree_node_t *nr = &bt->nodes[new_root];
        nr->is_leaf = 0U;
        nr->num_keys = 0U;
        nr->children[0] = old_root;
        nr->subtree_size = bt->nodes[old_root].subtree_size;
        bt->root = new_root;
        cs_btree_split_child(bt, new_root, 0U);
    }

    int32_t cur = bt->root;
    for (;;) {
        cs_btree_node_t *n = &bt->nodes[cur];
        n->subtree_size++;

        uint16_t i = 0;
        while (i < n->num_keys && n->keys[i] < x) {
            ++i;
        }
        if (i < n->num_keys && n->keys[i] == x) {
            n->counts[i]++;
            break;
        }
        if (n->is_leaf) {
            for (uint16_t j = n->num_keys; j > i; --j) {
                n->keys[j] = n->keys[j - 1U];
                n->counts[j] = n->counts[j - 1U];
            }
            n->keys[i] = x;
            n->counts[i] = 1U;
            n->num_keys++;
            break;
        }
        int32_t child_idx = n->children[i];
        if (bt->nodes[child_idx].num_keys == CS_BTREE_MAX_KEYS) {
            cs_btree_split_child(bt, cur, i);
            if (x == n->keys[i]) {
                n->counts[i]++;
                break;
            }
            if (x > n->keys[i]) {
                ++i;
            }
        }
        cur = n->children[i];
    }

    bt->total_count++;
    return CS_OK;
}

#endif /* CS_INTERNAL_H */
