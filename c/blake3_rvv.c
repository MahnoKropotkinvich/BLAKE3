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

// Transpose 4x4 matrix of uint32_t vectors
// Input: vecs[0-3] where each vec contains 4 elements
// After transpose: vecs[i][j] becomes vecs[j][i]
INLINE void transpose_vecs_rvv(vuint32m1_t vecs[4], size_t vl) {
  // Use scalar code for simplicity and correctness
  // TODO: optimize with vrgather if this becomes a bottleneck
  uint32_t temp[4][4];
  
  // Extract to temporary array
  __riscv_vse32_v_u32m1(temp[0], vecs[0], vl);
  __riscv_vse32_v_u32m1(temp[1], vecs[1], vl);
  __riscv_vse32_v_u32m1(temp[2], vecs[2], vl);
  __riscv_vse32_v_u32m1(temp[3], vecs[3], vl);
  
  // Transpose in place
  uint32_t transposed[4][4];
  for (size_t i = 0; i < vl && i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      transposed[i][j] = temp[j][i];
    }
  }
  
  // Load back
  vecs[0] = __riscv_vle32_v_u32m1(transposed[0], vl);
  vecs[1] = __riscv_vle32_v_u32m1(transposed[1], vl);
  vecs[2] = __riscv_vle32_v_u32m1(transposed[2], vl);
  vecs[3] = __riscv_vle32_v_u32m1(transposed[3], vl);
}

// Load and transpose message vectors using unit stride loads
// This replaces indexed loads with unit stride loads + transpose
INLINE void transpose_msg_vecs_rvv(const uint8_t *const *inputs,
                                    size_t block_offset,
                                    vuint32m1_t out[16],
                                    size_t vl) {
  // Load message words 0-3 from each input, then transpose
  vuint32m1_t rows[4];
  rows[0] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[0][block_offset + 0], vl);
  rows[1] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[1][block_offset + 0], vl);
  rows[2] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[2][block_offset + 0], vl);
  rows[3] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[3][block_offset + 0], vl);
  transpose_vecs_rvv(rows, vl);
  out[0] = rows[0]; out[1] = rows[1]; out[2] = rows[2]; out[3] = rows[3];
  
  // Load message words 4-7
  rows[0] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[0][block_offset + 16], vl);
  rows[1] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[1][block_offset + 16], vl);
  rows[2] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[2][block_offset + 16], vl);
  rows[3] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[3][block_offset + 16], vl);
  transpose_vecs_rvv(rows, vl);
  out[4] = rows[0]; out[5] = rows[1]; out[6] = rows[2]; out[7] = rows[3];
  
  // Load message words 8-11
  rows[0] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[0][block_offset + 32], vl);
  rows[1] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[1][block_offset + 32], vl);
  rows[2] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[2][block_offset + 32], vl);
  rows[3] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[3][block_offset + 32], vl);
  transpose_vecs_rvv(rows, vl);
  out[8] = rows[0]; out[9] = rows[1]; out[10] = rows[2]; out[11] = rows[3];
  
  // Load message words 12-15
  rows[0] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[0][block_offset + 48], vl);
  rows[1] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[1][block_offset + 48], vl);
  rows[2] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[2][block_offset + 48], vl);
  rows[3] = __riscv_vle32_v_u32m1((const uint32_t*)&inputs[3][block_offset + 48], vl);
  transpose_vecs_rvv(rows, vl);
  out[12] = rows[0]; out[13] = rows[1]; out[14] = rows[2]; out[15] = rows[3];
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

    // Load message words using unit stride loads + transpose
    // This replaces indexed loads for better cache locality
    size_t offset = block * BLAKE3_BLOCK_LEN;
    vuint32m1_t msg_vecs[16];
    transpose_msg_vecs_rvv(inputs, offset, msg_vecs, vl);
    
    vuint32m1_t m0 = msg_vecs[0];
    vuint32m1_t m1 = msg_vecs[1];
    vuint32m1_t m2 = msg_vecs[2];
    vuint32m1_t m3 = msg_vecs[3];
    vuint32m1_t m4 = msg_vecs[4];
    vuint32m1_t m5 = msg_vecs[5];
    vuint32m1_t m6 = msg_vecs[6];
    vuint32m1_t m7 = msg_vecs[7];
    vuint32m1_t m8 = msg_vecs[8];
    vuint32m1_t m9 = msg_vecs[9];
    vuint32m1_t m10 = msg_vecs[10];
    vuint32m1_t m11 = msg_vecs[11];
    vuint32m1_t m12 = msg_vecs[12];
    vuint32m1_t m13 = msg_vecs[13];
    vuint32m1_t m14 = msg_vecs[14];
    vuint32m1_t m15 = msg_vecs[15];

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

  // Transpose and store output using unit stride stores
  // This replaces strided stores for better performance
  vuint32m1_t h_vecs_low[4] = {h0, h1, h2, h3};
  vuint32m1_t h_vecs_high[4] = {h4, h5, h6, h7};
  transpose_vecs_rvv(h_vecs_low, vl);
  transpose_vecs_rvv(h_vecs_high, vl);
  
  // After transpose, h_vecs_low[i] contains first 4 words of output i
  // and h_vecs_high[i] contains last 4 words of output i
  // Output layout: [out0_low, out0_high, out1_low, out1_high, ...]
  __riscv_vse32_v_u32m1((uint32_t *)&out[0 * 16], h_vecs_low[0], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[1 * 16], h_vecs_high[0], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[2 * 16], h_vecs_low[1], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[3 * 16], h_vecs_high[1], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[4 * 16], h_vecs_low[2], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[5 * 16], h_vecs_high[2], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[6 * 16], h_vecs_low[3], vl);
  __riscv_vse32_v_u32m1((uint32_t *)&out[7 * 16], h_vecs_high[3], vl);
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
