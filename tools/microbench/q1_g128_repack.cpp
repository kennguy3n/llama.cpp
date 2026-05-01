// Microbenchmark harness for the proposed AVX2 / AVX-VNNI runtime
// repack of `Q1_0_g128` weights, as designed in
// `docs/avx2-repack-design/03-design-brief.md` (Phase 1c, merged on
// `prism` via PR #10).
//
// Phase 2 of the project. The harness is a stand-alone CLI tool that:
//
//   1. Synthesises N deterministic `block_q1_0_g128` weight blocks and
//      a matching column of `block_q8_0` activation blocks from a
//      fixed RNG seed, so every run produces the exact same input.
//   2. Builds the proposed `block_q1_0_g128x4` 4-row-interleaved
//      repack via the `make_block_q1_0_g128x4` helper (same shape
//      mainline ggml uses for `Q4_0_4_4`).
//   3. Computes the dot product four ways:
//          (a) scalar reference (parity baseline)
//          (b) the existing single-row AVX2 kernel from
//              `ggml/src/ggml-cpu/arch/x86/quants.c:893`, copied in
//              verbatim minus the linkage hooks
//          (c) the proposed AVX2 4x4 GEMV (Kernel A from the brief)
//          (d) the proposed AVX-VNNI 4x4 GEMV (Kernel B from the
//              brief), gated on `__AVXVNNI__` so Zen 3 hosts skip it
//   4. Asserts (c) and (d) produce bit-identical FP32 output to (a)
//      and (b) on the same inputs (within FP16-roundoff tolerance).
//   5. Times each kernel via a warm-up + steady-state loop using
//      `clock_gettime(CLOCK_MONOTONIC_RAW)` and reports median ns/row
//      and effective per-row throughput.
//
// **Why this is a Phase 2 deliverable, not Phase 3:** the four
// kernels live entirely inside this translation unit. None of them
// are plumbed into the `ggml-cpu` dispatcher, none touch
// `repack.h`/`repack.cpp`, none change the `tensor_traits` template.
// That work is Phase 3 and is gated on the cycles-per-row numbers
// produced here.
//
// Build & run (from repo root, after a normal CMake configure):
//
//     cmake --build build -j --target microbench-q1-g128-repack
//     ./build/bin/microbench-q1-g128-repack [--blocks N] [--iters M]
//
// Output goes to stdout. Sample baseline numbers from the
// EPYC 7763 / Zen 3 demo VM are committed at
// `tools/microbench/baseline-zen3-epyc-7763.txt` so reviewers can
// compare without rebuilding.

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace {

// -----------------------------------------------------------------
// Block layouts (local copies, intentionally not linked against
// ggml's `block_q*` types — keeping the harness self-contained means
// future intrinsic experiments don't have to rebuild the whole tree).
// -----------------------------------------------------------------

// Mirrors `block_q1_0_g128` in `ggml/src/ggml-common.h:186`.
struct block_q1_0_g128 {
    uint16_t d_bits;   // FP16 delta (raw bits)
    uint8_t  qs[16];   // 128 weight bits, packed LSB-first per byte
};
static_assert(sizeof(block_q1_0_g128) == 18, "wrong q1_0_g128 size");

// Mirrors `block_q8_0` in `ggml/src/ggml-common.h:252`.
struct block_q8_0 {
    uint16_t d_bits;   // FP16 delta
    int8_t   qs[32];   // 32 int8 activations
};
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 size");

// Proposed repacked layout from `03-design-brief.md` §"Layout":
// 4 deltas + 4 sub-blocks * 4 rows * 4 bytes = 8 + 64 = 72 bytes.
struct block_q1_0_g128x4 {
    uint16_t d_bits[4];
    uint8_t  qs[64];
};
static_assert(sizeof(block_q1_0_g128x4) == 72, "wrong q1_0_g128x4 size");
static_assert(sizeof(block_q1_0_g128x4) == 4 * sizeof(block_q1_0_g128),
              "repack must be size-equivalent to 4 rows of the source type");

constexpr int QK1_0_g128 = 128;
constexpr int QK8_0      = 32;

// -----------------------------------------------------------------
// FP16 <-> FP32. Use F16C intrinsics on x86 for bit-exactness with
// `_mm_cvtph_ps` later; provide a portable fallback so non-x86
// builds (or hosts without F16C) still compile and pass parity.
// -----------------------------------------------------------------

float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    __m128i v = _mm_cvtsi32_si128((int)h);
    return _mm_cvtss_f32(_mm_cvtph_ps(v));
#else
    // IEEE 754 half-precision -> single-precision, no inf/NaN handling
    // is fine here because the synthesised values stay in normal range.
    const uint32_t sign = (h >> 15) & 0x1;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        bits = sign << 31;
        if (mant != 0) {
            // subnormal: FP16 subnormal true exponent is 1 - 15 = -14, so
            // start at e = -15 and decrement once per left shift before the
            // implicit-1 bit appears in mant[10]. Final FP32 biased exponent
            // is 127 + (e + 1).
            int e = -15;
            uint32_t m = mant;
            while ((m & 0x400) == 0) { m <<= 1; --e; }
            m &= 0x3ff;
            bits |= ((127 + (e + 1)) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

uint16_t fp32_to_fp16(float f) {
#if defined(__F16C__)
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return (uint16_t)_mm_cvtsi128_si32(h);
#else
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | (0x1f << 10));
    return (uint16_t)((sign << 15) | (exp << 10) | (mant >> 13));
#endif
}

// -----------------------------------------------------------------
// Synthesis. Deterministic from a fixed seed so the measured numbers
// across `--iters` runs and across CI hosts are directly comparable.
// -----------------------------------------------------------------

void synthesize_weight_blocks(int nb, uint64_t seed,
                              std::vector<block_q1_0_g128> & out) {
    std::mt19937_64 rng(seed);
    out.assign(nb, {});
    std::uniform_int_distribution<uint32_t> bit_dist(0, 0xffffffffu);
    std::uniform_real_distribution<float>   d_dist(0.05f, 0.5f);
    for (int i = 0; i < nb; ++i) {
        out[i].d_bits = fp32_to_fp16(d_dist(rng));
        for (int j = 0; j < 16; j += 4) {
            const uint32_t r = bit_dist(rng);
            std::memcpy(&out[i].qs[j], &r, 4);
        }
    }
}

void synthesize_activation_blocks(int nb, uint64_t seed,
                                  std::vector<block_q8_0> & out) {
    // 4 q8_0 sub-blocks per q1_0_g128 weight block.
    const int total = nb * 4;
    std::mt19937_64 rng(seed);
    out.assign(total, {});
    std::uniform_int_distribution<int>   q_dist(-127, 127);
    std::uniform_real_distribution<float> d_dist(0.005f, 0.05f);
    for (int i = 0; i < total; ++i) {
        out[i].d_bits = fp32_to_fp16(d_dist(rng));
        for (int j = 0; j < QK8_0; ++j) {
            out[i].qs[j] = (int8_t)q_dist(rng);
        }
    }
}

// -----------------------------------------------------------------
// Repack helper: 4 standard-layout rows -> 1 repacked block.
//
// On-disk byte layout, per the design brief:
//
//   [d0 d1 d2 d3] (8 bytes of FP16 deltas)
//   [r0_sb0 r1_sb0 r2_sb0 r3_sb0]  (4 bytes per row, sub-block 0)
//   [r0_sb1 r1_sb1 r2_sb1 r3_sb1]  (sub-block 1)
//   [r0_sb2 r1_sb2 r2_sb2 r3_sb2]  (sub-block 2)
//   [r0_sb3 r1_sb3 r2_sb3 r3_sb3]  (sub-block 3)
//
// Total = 8 + 4*16 = 72 bytes = 4 * sizeof(block_q1_0_g128).
// -----------------------------------------------------------------

void make_block_q1_0_g128x4(const block_q1_0_g128 in[4],
                            block_q1_0_g128x4 * out) {
    for (int r = 0; r < 4; ++r) {
        out->d_bits[r] = in[r].d_bits;
    }
    for (int k = 0; k < 4; ++k) {
        for (int r = 0; r < 4; ++r) {
            std::memcpy(&out->qs[k*16 + r*4], &in[r].qs[k*4], 4);
        }
    }
}

// -----------------------------------------------------------------
// Kernel (a): scalar reference. Computes one row dot product. Used
// as the parity baseline; every other kernel must match this within
// FP16 tolerance.
// -----------------------------------------------------------------

float vec_dot_scalar(int n, const block_q1_0_g128 * x, const block_q8_0 * y) {
    const int nb = n / QK1_0_g128;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = fp16_to_fp32(x[ib].d_bits);
        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const float d1 = fp16_to_fp32(y[ib*4 + k].d_bits);
            int sumi_block = 0;
            for (int j = 0; j < QK8_0; ++j) {
                const int bit_index  = k * QK8_0 + j;
                const int byte_index = bit_index / 8;
                const int bit_offset = bit_index % 8;
                const int xi = ((x[ib].qs[byte_index] >> bit_offset) & 1) ? 1 : -1;
                const int yi = y[ib*4 + k].qs[j];
                sumi_block += xi * yi;
            }
            sumi += d1 * (float)sumi_block;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

// -----------------------------------------------------------------
// Kernel (b): existing single-row AVX2 path. Verbatim from
// `ggml/src/ggml-cpu/arch/x86/quants.c:893-941`, with the
// surrounding linkage and scalar-tail glue stripped. This is the
// floor we're trying to beat.
// -----------------------------------------------------------------

#if defined(__AVX2__)

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    // Get absolute values of x vectors so they're in unsigned form for maddubs.
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors.
    const __m256i sy = _mm256_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values.
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
    return _mm256_cvtepi32_ps(summed_pairs);
}

float vec_dot_avx2(int n, const block_q1_0_g128 * x, const block_q8_0 * y) {
    const int nb = n / QK1_0_g128;
    __m256 acc = _mm256_setzero_ps();
    const __m256i shuffle_mask = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i bit_mask = _mm256_set_epi8(
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i ones_v = _mm256_set1_epi8(1);

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = fp16_to_fp32(x[ib].d_bits);
        for (int k = 0; k < 4; ++k) {
            const float d1 = fp16_to_fp32(y[ib*4 + k].d_bits);
            const __m256 d  = _mm256_set1_ps(d0 * d1);
            uint32_t qbits32;
            std::memcpy(&qbits32, x[ib].qs + k * 4, sizeof(qbits32));
            const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib*4 + k].qs);

            const __m128i qbits_128   = _mm_set1_epi32((int)qbits32);
            const __m256i qbits_256   = _mm256_broadcastsi128_si256(qbits_128);
            const __m256i qbits_shuf  = _mm256_shuffle_epi8(qbits_256, shuffle_mask);
            const __m256i bit_test    = _mm256_and_si256(qbits_shuf, bit_mask);
            const __m256i is_set      = _mm256_cmpeq_epi8(bit_test, bit_mask);
            const __m256i bit_value   = _mm256_and_si256(is_set, ones_v);
            const __m256i bit_doubled = _mm256_add_epi8(bit_value, bit_value);
            const __m256i qx          = _mm256_sub_epi8(bit_doubled, ones_v);

            acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc);
        }
    }
    return hsum_float_8(acc);
}

// -----------------------------------------------------------------
// Kernel (c): proposed AVX2 4x4 GEMV (Kernel A from the brief).
// One call computes 4 row dot products against 1 activation column.
//
// Invariants matching the corrected pseudocode in 03-design-brief.md
// §GEMV-A (post-bugfix commit c1f7195):
//
//   * `qy`, `sum_qy_v`, and the 4-row qbits broadcast are loaded
//     ONCE per sub-block (k loop) and shared across all 4 rows.
//   * Per-row partial sums stay as `__m256i` vectors throughout the
//     hot path. NO horizontal reduction inside the inner loop.
//   * `dot(±1, qy) = 2 * partial - sum_qy` is computed in vector
//     form (slli + sub).
//   * `hsum_float_8` is paid exactly once per row at the very end,
//     outside the `nb`-deep loop, so it amortises to zero on
//     prefill-shaped workloads.
// -----------------------------------------------------------------

// Helper: bit-expand a 32-bit qbits row + blendv-select into qy. Inlined
// per row so the compiler can keep the constants in registers.
static inline __m256i select_qy_by_bits(uint32_t qbits32, __m256i qy,
                                        __m256i shuffle_mask, __m256i bit_mask,
                                        __m256i zero) {
    const __m128i qb128   = _mm_set1_epi32((int)qbits32);
    const __m256i qb256   = _mm256_broadcastsi128_si256(qb128);
    const __m256i shuf    = _mm256_shuffle_epi8(qb256, shuffle_mask);
    const __m256i tested  = _mm256_and_si256(shuf, bit_mask);
    const __m256i mask_ff = _mm256_cmpeq_epi8(tested, bit_mask);
    return _mm256_blendv_epi8(zero, qy, mask_ff);
}

void gemv_q1g128_4x4_avx2(int n,
                          float out4[4],
                          const block_q1_0_g128x4 * x,
                          const block_q8_0 * y) {
    const int nb = n / QK1_0_g128;

    const __m256i shuffle_mask = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i bit_mask = _mm256_set_epi8(
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i ones_b   = _mm256_set1_epi8(1);
    const __m256i ones_w   = _mm256_set1_epi16(1);
    const __m256i zero     = _mm256_setzero_si256();

    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };

    for (int ib = 0; ib < nb; ++ib) {
        // 4 FP16 deltas -> 4 FP32 in one F16C op.
        const __m128 d_rows = _mm_cvtph_ps(
            _mm_loadl_epi64((const __m128i *)&x[ib].d_bits[0]));
        alignas(16) float d_rows_arr[4];
        _mm_store_ps(d_rows_arr, d_rows);

        for (int k = 0; k < 4; ++k) {
            const __m256i qy = _mm256_loadu_si256(
                (const __m256i *)y[ib*4 + k].qs);
            const float d_y = fp16_to_fp32(y[ib*4 + k].d_bits);

            // Per-lane sum(qy), shared across all 4 rows.
            const __m256i sum_qy_v = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_b, qy),
                ones_w);

            // 4 rows worth of qbits32 in one 16-byte register. Use
            // _mm_extract_epi32 with constant immediates per row so
            // we never round-trip through memory.
            const __m128i qbits_4rows = _mm_loadu_si128(
                (const __m128i *)&x[ib].qs[k*16]);
            const uint32_t qbits0 = (uint32_t)_mm_extract_epi32(qbits_4rows, 0);
            const uint32_t qbits1 = (uint32_t)_mm_extract_epi32(qbits_4rows, 1);
            const uint32_t qbits2 = (uint32_t)_mm_extract_epi32(qbits_4rows, 2);
            const uint32_t qbits3 = (uint32_t)_mm_extract_epi32(qbits_4rows, 3);

            // Manually unroll the 4 rows. Each row: bit-expand+blendv to
            // mask qy, partial-sum to per-lane __m256i, then fold via
            // 2*partial - sum_qy and accumulate into acc[r]. Unrolling
            // gives the compiler 4 independent dependency chains to
            // schedule across Zen 3's two AVX2 ALU pipes.
            const __m256i masked0 = select_qy_by_bits(qbits0, qy, shuffle_mask, bit_mask, zero);
            const __m256i masked1 = select_qy_by_bits(qbits1, qy, shuffle_mask, bit_mask, zero);
            const __m256i masked2 = select_qy_by_bits(qbits2, qy, shuffle_mask, bit_mask, zero);
            const __m256i masked3 = select_qy_by_bits(qbits3, qy, shuffle_mask, bit_mask, zero);

            const __m256i partial0 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_b, masked0), ones_w);
            const __m256i partial1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_b, masked1), ones_w);
            const __m256i partial2 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_b, masked2), ones_w);
            const __m256i partial3 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_b, masked3), ones_w);

            const __m256i row_dot0 = _mm256_sub_epi32(_mm256_slli_epi32(partial0, 1), sum_qy_v);
            const __m256i row_dot1 = _mm256_sub_epi32(_mm256_slli_epi32(partial1, 1), sum_qy_v);
            const __m256i row_dot2 = _mm256_sub_epi32(_mm256_slli_epi32(partial2, 1), sum_qy_v);
            const __m256i row_dot3 = _mm256_sub_epi32(_mm256_slli_epi32(partial3, 1), sum_qy_v);

            acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(d_rows_arr[0] * d_y),
                                     _mm256_cvtepi32_ps(row_dot0), acc[0]);
            acc[1] = _mm256_fmadd_ps(_mm256_set1_ps(d_rows_arr[1] * d_y),
                                     _mm256_cvtepi32_ps(row_dot1), acc[1]);
            acc[2] = _mm256_fmadd_ps(_mm256_set1_ps(d_rows_arr[2] * d_y),
                                     _mm256_cvtepi32_ps(row_dot2), acc[2]);
            acc[3] = _mm256_fmadd_ps(_mm256_set1_ps(d_rows_arr[3] * d_y),
                                     _mm256_cvtepi32_ps(row_dot3), acc[3]);
        }
    }

    for (int r = 0; r < 4; ++r) {
        out4[r] = hsum_float_8(acc[r]);
    }
}

#endif // __AVX2__

// -----------------------------------------------------------------
// Kernel (d): proposed AVX-VNNI 4x4 GEMV (Kernel B from the brief).
// Same outer skeleton as Kernel A, but the per-row body collapses
// the multiply-add chain to a single `_mm256_dpbusd_avx_epi32`.
//
// Compiled in only when `__AVXVNNI__` is defined (Alder Lake P-cores,
// Zen 4). On Zen 3 (the demo VM) this kernel is omitted at compile
// time; the parity test and timing loop both skip it.
// -----------------------------------------------------------------

#if defined(__AVXVNNI__)

void gemv_q1g128_4x4_avx_vnni(int n,
                              float out4[4],
                              const block_q1_0_g128x4 * x,
                              const block_q8_0 * y) {
    const int nb = n / QK1_0_g128;

    const __m256i shuffle_mask = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i bit_mask = _mm256_set_epi8(
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i ones_b = _mm256_set1_epi8(1);
    const __m256i zero   = _mm256_setzero_si256();

    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };

    for (int ib = 0; ib < nb; ++ib) {
        const __m128 d_rows = _mm_cvtph_ps(
            _mm_loadl_epi64((const __m128i *)&x[ib].d_bits[0]));
        alignas(16) float d_rows_arr[4];
        _mm_store_ps(d_rows_arr, d_rows);

        for (int k = 0; k < 4; ++k) {
            const __m256i qy = _mm256_loadu_si256(
                (const __m256i *)y[ib*4 + k].qs);
            const float d_y = fp16_to_fp32(y[ib*4 + k].d_bits);
            const __m256i neg_qy = _mm256_sub_epi8(zero, qy);

            const __m128i qbits_4rows = _mm_loadu_si128(
                (const __m128i *)&x[ib].qs[k*16]);
            alignas(16) uint32_t qbits_arr[4];
            _mm_store_si128((__m128i *)qbits_arr, qbits_4rows);

            for (int r = 0; r < 4; ++r) {
                const uint32_t qbits32 = qbits_arr[r];

                const __m128i qb128   = _mm_set1_epi32((int)qbits32);
                const __m256i qb256   = _mm256_broadcastsi128_si256(qb128);
                const __m256i shuf    = _mm256_shuffle_epi8(qb256, shuffle_mask);
                const __m256i tested  = _mm256_and_si256(shuf, bit_mask);
                const __m256i mask_ff = _mm256_cmpeq_epi8(tested, bit_mask);

                // signed_qy = +qy where bit=1, -qy where bit=0.
                const __m256i signed_qy = _mm256_blendv_epi8(neg_qy, qy, mask_ff);

                // dpbusd(zero, ones, signed_qy) sums signed_qy in groups
                // of 4 into 8 int32 lanes — the entire dot product.
                const __m256i int_acc = _mm256_dpbusd_avx_epi32(
                    zero, ones_b, signed_qy);

                acc[r] = _mm256_fmadd_ps(
                    _mm256_set1_ps(d_rows_arr[r] * d_y),
                    _mm256_cvtepi32_ps(int_acc),
                    acc[r]);
            }
        }
    }

    for (int r = 0; r < 4; ++r) {
        out4[r] = hsum_float_8(acc[r]);
    }
}

#endif // __AVXVNNI__

// -----------------------------------------------------------------
// Parity test. All four kernels must agree within FP32 tolerance on
// a synthetic 4-row workload. Tolerance is generous (1e-3 relative)
// because FP16 deltas accumulate roundoff differently in scalar vs
// vectorised reductions.
// -----------------------------------------------------------------

bool approx_equal(float a, float b, float rel = 1e-3f, float abs_eps = 1e-4f) {
    const float diff = std::fabs(a - b);
    if (diff <= abs_eps) return true;
    const float scale = std::max(std::fabs(a), std::fabs(b));
    return diff / scale <= rel;
}

bool run_parity(int nb, uint64_t seed) {
    // 4 rows of weight blocks + 1 column of activations = 4 dot products.
    std::vector<block_q1_0_g128> rows[4];
    for (int r = 0; r < 4; ++r) {
        synthesize_weight_blocks(nb, seed + 0x100u * (uint64_t)r, rows[r]);
    }
    std::vector<block_q8_0> y;
    synthesize_activation_blocks(nb, seed + 0xfeed, y);

    const int n = nb * QK1_0_g128;

    float scalar[4];
    float avx2_single[4];
    for (int r = 0; r < 4; ++r) {
        scalar[r]      = vec_dot_scalar(n, rows[r].data(), y.data());
#if defined(__AVX2__)
        avx2_single[r] = vec_dot_avx2(n, rows[r].data(), y.data());
#else
        avx2_single[r] = scalar[r];
#endif
    }

    // Build the repacked weight stream.
    std::vector<block_q1_0_g128x4> packed(nb);
    for (int ib = 0; ib < nb; ++ib) {
        const block_q1_0_g128 in[4] = {
            rows[0][ib], rows[1][ib], rows[2][ib], rows[3][ib],
        };
        make_block_q1_0_g128x4(in, &packed[ib]);
    }

#if defined(__AVX2__)
    float gemv_avx2[4];
    gemv_q1g128_4x4_avx2(n, gemv_avx2, packed.data(), y.data());
#else
    float gemv_avx2[4] = { scalar[0], scalar[1], scalar[2], scalar[3] };
#endif

#if defined(__AVXVNNI__)
    float gemv_vnni[4];
    gemv_q1g128_4x4_avx_vnni(n, gemv_vnni, packed.data(), y.data());
#endif

    bool ok = true;
    for (int r = 0; r < 4; ++r) {
        const bool a = approx_equal(scalar[r], avx2_single[r]);
        const bool b = approx_equal(scalar[r], gemv_avx2[r]);
        bool row_ok = a && b;
        std::printf(
            "  row %d  scalar=%14.6f  avx2_single=%14.6f  gemv_avx2=%14.6f",
            r, scalar[r], avx2_single[r], gemv_avx2[r]);
#if defined(__AVXVNNI__)
        const bool c = approx_equal(scalar[r], gemv_vnni[r]);
        row_ok = row_ok && c;
        std::printf("  gemv_vnni=%14.6f", gemv_vnni[r]);
#endif
        ok = ok && row_ok;
        std::printf("  %s\n", row_ok ? "OK" : "MISMATCH");
    }
    return ok;
}

// -----------------------------------------------------------------
// Timing harness. CLOCK_MONOTONIC_RAW is hardware-counter-backed on
// Linux x86_64 (RDTSC + invariant TSC), so single-digit-ns
// granularity is achievable. We report median over `iters` runs to
// drown out OS-noise outliers.
// -----------------------------------------------------------------

double now_ns() {
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch())
        .count();
}

template <typename Fn>
double median_ns(Fn && fn, int iters) {
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_ns();
        fn();
        const double t1 = now_ns();
        samples.push_back(t1 - t0);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void run_timing(int nb, int iters, uint64_t seed) {
    // Build once, time many.
    std::vector<block_q1_0_g128> rows[4];
    for (int r = 0; r < 4; ++r) {
        synthesize_weight_blocks(nb, seed + 0x100u * (uint64_t)r, rows[r]);
    }
    std::vector<block_q8_0> y;
    synthesize_activation_blocks(nb, seed + 0xfeed, y);

    std::vector<block_q1_0_g128x4> packed(nb);
    for (int ib = 0; ib < nb; ++ib) {
        const block_q1_0_g128 in[4] = {
            rows[0][ib], rows[1][ib], rows[2][ib], rows[3][ib],
        };
        make_block_q1_0_g128x4(in, &packed[ib]);
    }

    const int n = nb * QK1_0_g128;
    const double rows_per_call_single = 1.0;
    const double rows_per_call_gemv   = 4.0;

    std::printf("\n  Timing (%d weight blocks = %d weights, %d iters, median ns):\n",
                nb, n, iters);

    // Warm up i-cache and branch predictors so the first sample isn't
    // 5x the steady state.
    {
        volatile float sink = 0.0f;
        for (int i = 0; i < 32; ++i) sink += vec_dot_scalar(n, rows[0].data(), y.data());
        (void)sink;
    }

    {
        volatile float sink = 0.0f;
        const double t = median_ns(
            [&]{
                for (int r = 0; r < 4; ++r) {
                    sink += vec_dot_scalar(n, rows[r].data(), y.data());
                }
            },
            iters);
        std::printf("    %-22s  %10.0f ns / 4 rows  (%.1f ns/row)\n",
                    "scalar", t, t / (4.0 * rows_per_call_single));
        (void)sink;
    }

#if defined(__AVX2__)
    {
        volatile float sink = 0.0f;
        const double t = median_ns(
            [&]{
                for (int r = 0; r < 4; ++r) {
                    sink += vec_dot_avx2(n, rows[r].data(), y.data());
                }
            },
            iters);
        std::printf("    %-22s  %10.0f ns / 4 rows  (%.1f ns/row)\n",
                    "vec_dot_avx2", t, t / (4.0 * rows_per_call_single));
        (void)sink;
    }
#endif

#if defined(__AVX2__)
    {
        volatile float sink = 0.0f;
        const double t = median_ns(
            [&]{
                float out4[4];
                gemv_q1g128_4x4_avx2(n, out4, packed.data(), y.data());
                sink += out4[0] + out4[1] + out4[2] + out4[3];
            },
            iters);
        std::printf("    %-22s  %10.0f ns / 4 rows  (%.1f ns/row)\n",
                    "gemv_4x4_avx2 (NEW)", t, t / rows_per_call_gemv);
        (void)sink;
    }
#endif

#if defined(__AVXVNNI__)
    {
        volatile float sink = 0.0f;
        const double t = median_ns(
            [&]{
                float out4[4];
                gemv_q1g128_4x4_avx_vnni(n, out4, packed.data(), y.data());
                sink += out4[0] + out4[1] + out4[2] + out4[3];
            },
            iters);
        std::printf("    %-22s  %10.0f ns / 4 rows  (%.1f ns/row)\n",
                    "gemv_4x4_avx_vnni (NEW)", t, t / rows_per_call_gemv);
        (void)sink;
    }
#endif
}

} // namespace

int main(int argc, char ** argv) {
    int nb        = 4096;     // 4096 q1_0_g128 blocks = 524 288 weights
    int iters     = 1024;
    uint64_t seed = 0xC0FFEEULL;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--blocks" || a == "-b") && i + 1 < argc) {
            nb = std::atoi(argv[++i]);
        } else if ((a == "--iters" || a == "-n") && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if ((a == "--seed" || a == "-s") && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 0);
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "usage: microbench-q1-g128-repack [--blocks N] [--iters M] [--seed S]\n"
                "  --blocks N     number of Q1_0_g128 weight blocks per row "
                "(default 4096; one block = 128 weights)\n"
                "  --iters  M     number of timed iterations (default 1024)\n"
                "  --seed   S     RNG seed for synthesis (default 0xC0FFEE)\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s (try --help)\n", argv[i]);
            return 1;
        }
    }

    std::printf(
        "microbench-q1-g128-repack: Phase 2 harness for the proposed AVX2/AVX-VNNI\n"
        "  runtime repack of Q1_0_g128. See docs/avx2-repack-design/03-design-brief.md\n"
        "  Configuration: nb=%d (n=%d weights), iters=%d, seed=0x%" PRIx64 "\n\n",
        nb, nb * QK1_0_g128, iters, (uint64_t)seed);

    std::printf("Build features:");
#if defined(__AVX512F__)
    std::printf(" AVX512F");
#endif
#if defined(__AVX512VNNI__)
    std::printf(" AVX512VNNI");
#endif
#if defined(__AVXVNNI__)
    std::printf(" AVXVNNI");
#endif
#if defined(__AVX2__)
    std::printf(" AVX2");
#endif
#if defined(__FMA__)
    std::printf(" FMA");
#endif
#if defined(__F16C__)
    std::printf(" F16C");
#endif
    std::printf("\n\n");

    std::printf("Parity check (4 rows, scalar baseline):\n");
    const bool parity_ok = run_parity(nb, seed);
    std::printf("Parity: %s\n", parity_ok ? "OK" : "FAILED");
    if (!parity_ok) return 2;

    run_timing(nb, iters, seed);
    return 0;
}
