#include "blake3_impl.h"
#include <riscv_vector.h>

void blake3_compress_in_place_portable(uint32_t cv[8],
                                       const uint8_t block[BLAKE3_BLOCK_LEN],
                                       uint8_t block_len, uint64_t counter,
                                       uint8_t flags);

INLINE vuint32m1_t add(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vadd_vv_u32m1(a, b, vl);
}

INLINE vuint32m1_t xor(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vxor_vv_u32m1(a, b, vl);
}

INLINE vuint32m1_t set1(uint32_t x, size_t vl) {
  return __riscv_vmv_v_x_u32m1(x, vl);
}

INLINE vuint32m1_t rot16(vuint32m1_t x, size_t vl) {
  return __riscv_vor_vv_u32m1(__riscv_vsrl_vx_u32m1(x, 16, vl),
                               __riscv_vsll_vx_u32m1(x, 16, vl), vl);
}

INLINE vuint32m1_t rot12(vuint32m1_t x, size_t vl) {
  return __riscv_vor_vv_u32m1(__riscv_vsrl_vx_u32m1(x, 12, vl),
                               __riscv_vsll_vx_u32m1(x, 20, vl), vl);
}

INLINE vuint32m1_t rot8(vuint32m1_t x, size_t vl) {
  return __riscv_vor_vv_u32m1(__riscv_vsrl_vx_u32m1(x, 8, vl),
                               __riscv_vsll_vx_u32m1(x, 24, vl), vl);
}

INLINE vuint32m1_t rot7(vuint32m1_t x, size_t vl) {
  return __riscv_vor_vv_u32m1(__riscv_vsrl_vx_u32m1(x, 7, vl),
                               __riscv_vsll_vx_u32m1(x, 25, vl), vl);
}

INLINE void g(vuint32m1_t *a, vuint32m1_t *b, vuint32m1_t *c, vuint32m1_t *d,
              vuint32m1_t mx, vuint32m1_t my, size_t vl) {
  *a = add(*a, add(*b, mx, vl), vl);
  *d = rot16(xor(*d, *a, vl), vl);
  *c = add(*c, *d, vl);
  *b = rot12(xor(*b, *c, vl), vl);
  *a = add(*a, add(*b, my, vl), vl);
  *d = rot8(xor(*d, *a, vl), vl);
  *c = add(*c, *d, vl);
  *b = rot7(xor(*b, *c, vl), vl);
}

INLINE vuint32m1_t get_msg(vuint32m1_t m0, vuint32m1_t m1, vuint32m1_t m2, vuint32m1_t m3,
                            vuint32m1_t m4, vuint32m1_t m5, vuint32m1_t m6, vuint32m1_t m7,
                            vuint32m1_t m8, vuint32m1_t m9, vuint32m1_t m10, vuint32m1_t m11,
                            vuint32m1_t m12, vuint32m1_t m13, vuint32m1_t m14, vuint32m1_t m15,
                            size_t idx) {
  switch(idx) {
    case 0: return m0;
    case 1: return m1;
    case 2: return m2;
    case 3: return m3;
    case 4: return m4;
    case 5: return m5;
    case 6: return m6;
    case 7: return m7;
    case 8: return m8;
    case 9: return m9;
    case 10: return m10;
    case 11: return m11;
    case 12: return m12;
    case 13: return m13;
    case 14: return m14;
    default: return m15;
  }
}

INLINE void round_fn(vuint32m1_t *v0, vuint32m1_t *v1, vuint32m1_t *v2, vuint32m1_t *v3,
                     vuint32m1_t *v4, vuint32m1_t *v5, vuint32m1_t *v6, vuint32m1_t *v7,
                     vuint32m1_t *v8, vuint32m1_t *v9, vuint32m1_t *v10, vuint32m1_t *v11,
                     vuint32m1_t *v12, vuint32m1_t *v13, vuint32m1_t *v14, vuint32m1_t *v15,
                     vuint32m1_t m0, vuint32m1_t m1, vuint32m1_t m2, vuint32m1_t m3,
                     vuint32m1_t m4, vuint32m1_t m5, vuint32m1_t m6, vuint32m1_t m7,
                     vuint32m1_t m8, vuint32m1_t m9, vuint32m1_t m10, vuint32m1_t m11,
                     vuint32m1_t m12, vuint32m1_t m13, vuint32m1_t m14, vuint32m1_t m15,
                     size_t r, size_t vl) {
  g(v0, v4, v8, v12, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][0]),
                     get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][1]), vl);
  g(v1, v5, v9, v13, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][2]),
                     get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][3]), vl);
  g(v2, v6, v10, v14, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][4]),
                      get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][5]), vl);
  g(v3, v7, v11, v15, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][6]),
                      get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][7]), vl);
  g(v0, v5, v10, v15, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][8]),
                      get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][9]), vl);
  g(v1, v6, v11, v12, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][10]),
                      get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][11]), vl);
  g(v2, v7, v8, v13, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][12]),
                     get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][13]), vl);
  g(v3, v4, v9, v14, get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][14]),
                     get_msg(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, MSG_SCHEDULE[r][15]), vl);
}

// Hash vl inputs in parallel using RVV
INLINE void blake3_hash_vl_rvv(const uint8_t *const *inputs, size_t vl,
                                size_t blocks, const uint32_t key[8],
                                uint64_t counter, bool increment_counter,
                                uint8_t flags, uint8_t flags_start,
                                uint8_t flags_end, uint8_t *out) {
  vuint32m1_t h0 = set1(key[0], vl);
  vuint32m1_t h1 = set1(key[1], vl);
  vuint32m1_t h2 = set1(key[2], vl);
  vuint32m1_t h3 = set1(key[3], vl);
  vuint32m1_t h4 = set1(key[4], vl);
  vuint32m1_t h5 = set1(key[5], vl);
  vuint32m1_t h6 = set1(key[6], vl);
  vuint32m1_t h7 = set1(key[7], vl);

  uint32_t low_vals[16];
  uint32_t high_vals[16];
  for (size_t i = 0; i < vl && i < 16; i++) {
    uint64_t c = counter + (increment_counter ? i : 0);
    low_vals[i] = (uint32_t)c;
    high_vals[i] = (uint32_t)(c >> 32);
  }
  vuint32m1_t counter_low = __riscv_vle32_v_u32m1(low_vals, vl);
  vuint32m1_t counter_high = __riscv_vle32_v_u32m1(high_vals, vl);

  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }

    // Load message words using scalar loads + unit stride vector load
    // This replaces indexed loads to test if it's faster
    size_t offset = block * BLAKE3_BLOCK_LEN;
    const uint32_t *input_ptrs[16];
    for (size_t i = 0; i < vl && i < 16; i++) {
      input_ptrs[i] = (const uint32_t *)&inputs[i][offset];
    }
    
    // Load each message word using scalar loads, then create vectors
    uint32_t m0_vals[16], m1_vals[16], m2_vals[16], m3_vals[16];
    uint32_t m4_vals[16], m5_vals[16], m6_vals[16], m7_vals[16];
    uint32_t m8_vals[16], m9_vals[16], m10_vals[16], m11_vals[16];
    uint32_t m12_vals[16], m13_vals[16], m14_vals[16], m15_vals[16];
    
    for (size_t i = 0; i < vl && i < 16; i++) {
      m0_vals[i] = input_ptrs[i][0];
      m1_vals[i] = input_ptrs[i][1];
      m2_vals[i] = input_ptrs[i][2];
      m3_vals[i] = input_ptrs[i][3];
      m4_vals[i] = input_ptrs[i][4];
      m5_vals[i] = input_ptrs[i][5];
      m6_vals[i] = input_ptrs[i][6];
      m7_vals[i] = input_ptrs[i][7];
      m8_vals[i] = input_ptrs[i][8];
      m9_vals[i] = input_ptrs[i][9];
      m10_vals[i] = input_ptrs[i][10];
      m11_vals[i] = input_ptrs[i][11];
      m12_vals[i] = input_ptrs[i][12];
      m13_vals[i] = input_ptrs[i][13];
      m14_vals[i] = input_ptrs[i][14];
      m15_vals[i] = input_ptrs[i][15];
    }
    
    vuint32m1_t m0 = __riscv_vle32_v_u32m1(m0_vals, vl);
    vuint32m1_t m1 = __riscv_vle32_v_u32m1(m1_vals, vl);
    vuint32m1_t m2 = __riscv_vle32_v_u32m1(m2_vals, vl);
    vuint32m1_t m3 = __riscv_vle32_v_u32m1(m3_vals, vl);
    vuint32m1_t m4 = __riscv_vle32_v_u32m1(m4_vals, vl);
    vuint32m1_t m5 = __riscv_vle32_v_u32m1(m5_vals, vl);
    vuint32m1_t m6 = __riscv_vle32_v_u32m1(m6_vals, vl);
    vuint32m1_t m7 = __riscv_vle32_v_u32m1(m7_vals, vl);
    vuint32m1_t m8 = __riscv_vle32_v_u32m1(m8_vals, vl);
    vuint32m1_t m9 = __riscv_vle32_v_u32m1(m9_vals, vl);
    vuint32m1_t m10 = __riscv_vle32_v_u32m1(m10_vals, vl);
    vuint32m1_t m11 = __riscv_vle32_v_u32m1(m11_vals, vl);
    vuint32m1_t m12 = __riscv_vle32_v_u32m1(m12_vals, vl);
    vuint32m1_t m13 = __riscv_vle32_v_u32m1(m13_vals, vl);
    vuint32m1_t m14 = __riscv_vle32_v_u32m1(m14_vals, vl);
    vuint32m1_t m15 = __riscv_vle32_v_u32m1(m15_vals, vl);

    vuint32m1_t v0 = h0;
    vuint32m1_t v1 = h1;
    vuint32m1_t v2 = h2;
    vuint32m1_t v3 = h3;
    vuint32m1_t v4 = h4;
    vuint32m1_t v5 = h5;
    vuint32m1_t v6 = h6;
    vuint32m1_t v7 = h7;
    vuint32m1_t v8 = set1(IV[0], vl);
    vuint32m1_t v9 = set1(IV[1], vl);
    vuint32m1_t v10 = set1(IV[2], vl);
    vuint32m1_t v11 = set1(IV[3], vl);
    vuint32m1_t v12 = counter_low;
    vuint32m1_t v13 = counter_high;
    vuint32m1_t v14 = set1((uint32_t)BLAKE3_BLOCK_LEN, vl);
    vuint32m1_t v15 = set1((uint32_t)block_flags, vl);

    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 0, vl);
    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 1, vl);
    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 2, vl);
    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 3, vl);
    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 4, vl);
    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 5, vl);
    round_fn(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8, &v9, &v10, &v11, &v12, &v13, &v14, &v15,
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, 6, vl);

    h0 = xor(v0, v8, vl);
    h1 = xor(v1, v9, vl);
    h2 = xor(v2, v10, vl);
    h3 = xor(v3, v11, vl);
    h4 = xor(v4, v12, vl);
    h5 = xor(v5, v13, vl);
    h6 = xor(v6, v14, vl);
    h7 = xor(v7, v15, vl);

    block_flags = flags;
  }

  // Store output using scalar stores
  // This replaces strided stores to test if it's faster
  uint32_t h0_vals[16], h1_vals[16], h2_vals[16], h3_vals[16];
  uint32_t h4_vals[16], h5_vals[16], h6_vals[16], h7_vals[16];
  
  __riscv_vse32_v_u32m1(h0_vals, h0, vl);
  __riscv_vse32_v_u32m1(h1_vals, h1, vl);
  __riscv_vse32_v_u32m1(h2_vals, h2, vl);
  __riscv_vse32_v_u32m1(h3_vals, h3, vl);
  __riscv_vse32_v_u32m1(h4_vals, h4, vl);
  __riscv_vse32_v_u32m1(h5_vals, h5, vl);
  __riscv_vse32_v_u32m1(h6_vals, h6, vl);
  __riscv_vse32_v_u32m1(h7_vals, h7, vl);
  
  for (size_t i = 0; i < vl && i < 16; i++) {
    uint32_t *out_ptr = (uint32_t *)&out[i * BLAKE3_OUT_LEN];
    out_ptr[0] = h0_vals[i];
    out_ptr[1] = h1_vals[i];
    out_ptr[2] = h2_vals[i];
    out_ptr[3] = h3_vals[i];
    out_ptr[4] = h4_vals[i];
    out_ptr[5] = h5_vals[i];
    out_ptr[6] = h6_vals[i];
    out_ptr[7] = h7_vals[i];
  }
}

void blake3_hash_many_rvv(const uint8_t *const *inputs, size_t num_inputs,
                          size_t blocks, const uint32_t key[8],
                          uint64_t counter, bool increment_counter,
                          uint8_t flags, uint8_t flags_start,
                          uint8_t flags_end, uint8_t *out) {
  size_t vl = __riscv_vsetvlmax_e32m1();
  
  while (num_inputs >= vl) {
    blake3_hash_vl_rvv(inputs, vl, blocks, key, counter, increment_counter,
                       flags, flags_start, flags_end, out);
    
    if (increment_counter) {
      counter += vl;
    }
    inputs += vl;
    num_inputs -= vl;
    out += vl * BLAKE3_OUT_LEN;
  }

  while (num_inputs > 0) {
    uint32_t cv[8];
    memcpy(cv, key, BLAKE3_KEY_LEN);
    uint8_t block_flags = flags | flags_start;

    const uint8_t *input = inputs[0];
    for (size_t block = 0; block < blocks; block++) {
      if (block + 1 == blocks) {
        block_flags |= flags_end;
      }
      blake3_compress_in_place_portable(cv, input, BLAKE3_BLOCK_LEN, counter,
                                        block_flags);
      input += BLAKE3_BLOCK_LEN;
      block_flags = flags;
    }

    memcpy(out, cv, BLAKE3_OUT_LEN);

    if (increment_counter) {
      counter += 1;
    }
    inputs += 1;
    num_inputs -= 1;
    out += BLAKE3_OUT_LEN;
  }
}
