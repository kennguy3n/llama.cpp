// Parity test for the Phase 3 Q1_0_g128 4-row repack GEMV/GEMM kernels.
//
// Background:
//   Phase 2 (PR #11) added an offline microbench harness for the proposed
//   AVX2 / AVX-VNNI repack kernels. Phase 3 (PR #12) ports the AVX2 path
//   into `ggml/src/ggml-cpu/arch/x86/repack.cpp` and wires it into the
//   runtime repack dispatcher in `repack.cpp`.
//
//   This test verifies that the production GEMV and GEMM kernels (the
//   ones that actually run at inference time on AVX2-only hosts) produce
//   FP32 output bit-identical (within FP16 quantisation noise) to the
//   single-row `vec_dot` already in mainline.
//
//   The first iteration of this test only exercised GEMV (`nr = 1`).
//   Devin Review on PR #12 caught a silent-corruption bug in the GEMM
//   path that the GEMV-only test missed: GEMM is invoked by
//   `forward_mul_mat_one_chunk` whenever `nrows > 3`, with src1
//   pre-packed as `block_q8_0x4` (4 activation rows interleaved). The
//   fixed GEMM unpacks rows on the fly; this test now covers both
//   nr=1 (GEMV) and nr=4/8 (GEMM) to keep that fix from regressing.
//
// Test design:
//   * 4 weight rows of `n_weights = 4 * QK1_0_g128 = 512` weights each,
//     drawn from a deterministic pseudo-random generator. Quantised
//     once as Q1_0_g128 (single-row) and once via the manual repack
//     helper into the `block_q1_0_g128x4` layout.
//   * For GEMV: 1 activation row of n_weights floats, quantised as
//     plain block_q8_0.
//   * For GEMM: nr activation rows of n_weights floats, quantised as
//     block_q8_0 (single-row) for the reference, then packed into
//     `block_q8_0x4` groups (mirror of `ggml_quantize_mat_q8_0_4x4`)
//     for the kernel input.
//   * Reference output: single-row `vec_dot` from
//     `ggml_get_type_traits_cpu(GGML_TYPE_Q1_0_g128)`.
//   * Test output: `ggml_gemv_q1_0_g128_4x4_q8_0` (nr=1) or
//     `ggml_gemm_q1_0_g128_4x4_q8_0` (nr>=4).
//   * Pass criterion: per-element absolute error <= 1e-3 of the
//     per-row magnitude (FP16 round-trip + FMA reordering noise).
//
// The struct layouts below MUST match `block_q1_0_g128x4` and
// `block_q8_0x4` in `ggml/src/ggml-cpu/repack.h`. They are duplicated
// here only to avoid adding the internal header to the test include path.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int K_QK1_0_G128 = 128;
constexpr int K_QK8_0      = 32;

// Mirror of `block_q1_0_g128x4` from `ggml/src/ggml-cpu/repack.h`.
struct block_q1_0_g128x4_test {
    uint16_t d[4];
    uint8_t  qs[64];
};
static_assert(sizeof(block_q1_0_g128x4_test) == 72,
              "test layout must match ggml/src/ggml-cpu/repack.h");

// Mirror of `block_q1_0_g128` from `ggml/src/ggml-common.h` — single-row
// layout the dispatcher repacks from.
struct block_q1_0_g128_test {
    uint16_t d;
    uint8_t  qs[K_QK1_0_G128 / 8];
};
static_assert(sizeof(block_q1_0_g128_test) == 18,
              "test single-row layout must match ggml-common.h");

// Mirror of `block_q8_0` from `ggml/src/ggml-common.h`.
struct block_q8_0_test {
    uint16_t d;
    int8_t   qs[K_QK8_0];
};
static_assert(sizeof(block_q8_0_test) == 34,
              "test q8_0 layout must match ggml-common.h");

// Mirror of `block<8, 4>` (= `block_q8_0x4`) from
// `ggml/src/ggml-cpu/repack.h`. 4 deltas + 4 rows × 32 q8 values
// interleaved with `blck_size_interleave = 4`.
struct block_q8_0x4_test {
    uint16_t d[4];
    int8_t   qs[K_QK8_0 * 4];
};
static_assert(sizeof(block_q8_0x4_test) == 8 + K_QK8_0 * 4,
              "test q8_0x4 layout must match repack.h");

// Forward-declare the production GEMV / GEMM. Defined in
// `ggml/src/ggml-cpu/arch/x86/repack.cpp` (or `repack.cpp` for non-x86).
extern "C" void ggml_gemv_q1_0_g128_4x4_q8_0(int n, float * s, size_t bs,
                                             const void * vx, const void * vy,
                                             int nr, int nc);
extern "C" void ggml_gemm_q1_0_g128_4x4_q8_0(int n, float * s, size_t bs,
                                             const void * vx, const void * vy,
                                             int nr, int nc);

// Repack 4 single-row blocks into one x4 block, mirroring
// `make_block_q1_0_g128x4` in `repack.cpp`.
void repack_4_rows_to_x4(const block_q1_0_g128_test in[4],
                         block_q1_0_g128x4_test * out) {
    for (int r = 0; r < 4; r++) {
        out->d[r] = in[r].d;
    }
    for (int k = 0; k < 4; k++) {
        for (int r = 0; r < 4; r++) {
            std::memcpy(&out->qs[k * 16 + r * 4], &in[r].qs[k * 4], 4);
        }
    }
}

// Pack 4 quantised activation rows (each `nb` `block_q8_0`) into a
// stream of `nb` `block_q8_0x4`, mirroring the encoder in
// `ggml_quantize_mat_q8_0_4x4_generic` (`repack.cpp:51`).
//   blck_size_interleave = 4
//   src_offset = (j / 16) * 4 + (j % 4)
//   src_id     = (j % 16) / 4
void pack_4_rows_to_q8_0x4(const std::vector<block_q8_0_test> rows[4],
                           int nb, block_q8_0x4_test * out) {
    for (int ib = 0; ib < nb; ib++) {
        for (int r = 0; r < 4; r++) {
            out[ib].d[r] = rows[r][ib].d;
        }
        for (int j = 0; j < K_QK8_0 * 4; j++) {
            const int src_id     = (j % 16) / 4;
            const int src_offset = (j / 16) * 4 + (j % 4);
            out[ib].qs[j]        = rows[src_id][ib].qs[src_offset];
        }
    }
}

float max_abs_value(const std::vector<float> & v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

struct test_context {
    const ggml_type_traits_cpu * traits_q1   = nullptr;
    const ggml_type_traits_cpu * traits_q8_0 = nullptr;

    std::vector<std::vector<float>>            w;          // [n_w_rows][n_weights]
    std::vector<block_q1_0_g128_test>          q1_rows;    // [n_w_rows * n_blocks]
    std::vector<block_q1_0_g128x4_test>        packed_w;   // [n_blocks]
    int n_w_rows  = 0;
    int n_weights = 0;
    int n_blocks  = 0; // QK1_0_g128 blocks per row
    int n_q8      = 0; // QK8_0 blocks per row
};

bool prepare_weights(test_context & ctx, std::mt19937 & rng) {
    ctx.traits_q1   = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0_g128);
    ctx.traits_q8_0 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!ctx.traits_q1 || !ctx.traits_q1->from_float || !ctx.traits_q1->vec_dot ||
        !ctx.traits_q8_0 || !ctx.traits_q8_0->from_float) {
        printf("FAIL: cpu type traits missing for Q1_0_g128 / Q8_0\n");
        return false;
    }
    if (ctx.traits_q1->vec_dot_type != GGML_TYPE_Q8_0) {
        printf("FAIL: expected Q1_0_g128 vec_dot_type == Q8_0, got %d\n",
               (int) ctx.traits_q1->vec_dot_type);
        return false;
    }

    ctx.n_w_rows  = 4;
    ctx.n_blocks  = 4;
    ctx.n_weights = K_QK1_0_G128 * ctx.n_blocks;
    ctx.n_q8      = ctx.n_weights / K_QK8_0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    ctx.w.assign(ctx.n_w_rows, std::vector<float>(ctx.n_weights));
    for (int r = 0; r < ctx.n_w_rows; r++) {
        for (int i = 0; i < ctx.n_weights; i++) ctx.w[r][i] = dist(rng);
    }

    ctx.q1_rows.assign((size_t) ctx.n_w_rows * ctx.n_blocks, {});
    for (int r = 0; r < ctx.n_w_rows; r++) {
        ctx.traits_q1->from_float(ctx.w[r].data(),
                                  ctx.q1_rows.data() + r * ctx.n_blocks,
                                  ctx.n_weights);
    }

    ctx.packed_w.assign(ctx.n_blocks, {});
    {
        block_q1_0_g128_test tmp4[4];
        for (int ib = 0; ib < ctx.n_blocks; ib++) {
            for (int r = 0; r < ctx.n_w_rows; r++) {
                tmp4[r] = ctx.q1_rows[r * ctx.n_blocks + ib];
            }
            repack_4_rows_to_x4(tmp4, &ctx.packed_w[ib]);
        }
    }
    return true;
}

// Run a single GEMV (nr=1) test case. Output layout: 1 row of n_w_rows
// floats. Compares against single-row vec_dot.
bool run_gemv_test(const test_context & ctx, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> y(ctx.n_weights);
    for (int i = 0; i < ctx.n_weights; i++) y[i] = dist(rng);

    std::vector<block_q8_0_test> q8(ctx.n_q8);
    ctx.traits_q8_0->from_float(y.data(), q8.data(), ctx.n_weights);

    std::vector<float> ref(ctx.n_w_rows, 0.0f);
    for (int r = 0; r < ctx.n_w_rows; r++) {
        ctx.traits_q1->vec_dot(ctx.n_weights,
                               &ref[r], /* bs = */ 0,
                               ctx.q1_rows.data() + r * ctx.n_blocks, /* bx = */ 0,
                               q8.data(), /* by = */ 0,
                               /* nrc = */ 1);
    }

    std::vector<float> got(ctx.n_w_rows, 0.0f);
    ggml_gemv_q1_0_g128_4x4_q8_0(ctx.n_weights,
                                 got.data(), /* bs = */ 0,
                                 ctx.packed_w.data(), q8.data(),
                                 /* nr = */ 1, /* nc = */ ctx.n_w_rows);

    const float ref_mag = std::max(max_abs_value(ref), 1e-6f);
    const float tol     = 1e-3f * ref_mag + 1e-4f;
    int failed = 0;
    for (int r = 0; r < ctx.n_w_rows; r++) {
        const float diff = std::fabs(got[r] - ref[r]);
        const bool  ok   = diff <= tol;
        printf("  GEMV row %d: ref=% .6f got=% .6f diff=% .6e %s\n",
               r, (double) ref[r], (double) got[r], (double) diff,
               ok ? "OK" : "FAIL");
        failed += !ok;
    }
    if (failed) {
        printf("  GEMV: %d/%d rows out of tolerance\n", failed, ctx.n_w_rows);
    }
    return failed == 0;
}

// Run a GEMM test case with `nr` activation rows. Output is laid out
// as a `nr × n_w_rows` matrix with the same `bs` (float-element row
// stride) the production dispatcher passes; we choose `bs = n_w_rows`.
bool run_gemm_test(const test_context & ctx, int nr, std::mt19937 & rng) {
    assert(nr % 4 == 0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Synthetic activation rows.
    std::vector<std::vector<float>> y(nr, std::vector<float>(ctx.n_weights));
    for (int row = 0; row < nr; row++) {
        for (int i = 0; i < ctx.n_weights; i++) y[row][i] = dist(rng);
    }

    // Per-row Q8_0 quantisation (single-row layout, used for both the
    // reference vec_dot and as input to the q8_0x4 packer).
    std::vector<std::vector<block_q8_0_test>> q8_per_row(
        nr, std::vector<block_q8_0_test>(ctx.n_q8));
    for (int row = 0; row < nr; row++) {
        ctx.traits_q8_0->from_float(y[row].data(),
                                    q8_per_row[row].data(),
                                    ctx.n_weights);
    }

    // Pack into block_q8_0x4 groups, matching what
    // `ggml_quantize_mat_q8_0_4x4` produces at runtime.
    const int n_groups = nr / 4;
    std::vector<block_q8_0x4_test> q8_pack((size_t) n_groups * ctx.n_q8);
    for (int g = 0; g < n_groups; g++) {
        const std::vector<block_q8_0_test> * rows[4] = {
            &q8_per_row[g * 4 + 0], &q8_per_row[g * 4 + 1],
            &q8_per_row[g * 4 + 2], &q8_per_row[g * 4 + 3],
        };
        std::vector<block_q8_0_test> rows_arr[4] = {
            *rows[0], *rows[1], *rows[2], *rows[3],
        };
        pack_4_rows_to_q8_0x4(rows_arr, ctx.n_q8,
                              q8_pack.data() + (size_t) g * ctx.n_q8);
    }

    // Reference: nr × n_w_rows single-row vec_dots.
    const int    bs  = ctx.n_w_rows;
    const size_t out = (size_t) nr * bs;
    std::vector<float> ref(out, 0.0f);
    for (int row = 0; row < nr; row++) {
        for (int r = 0; r < ctx.n_w_rows; r++) {
            ctx.traits_q1->vec_dot(ctx.n_weights,
                                   &ref[(size_t) row * bs + r], /* bs = */ 0,
                                   ctx.q1_rows.data() + r * ctx.n_blocks, /* bx = */ 0,
                                   q8_per_row[row].data(), /* by = */ 0,
                                   /* nrc = */ 1);
        }
    }

    // Test: one GEMM call.
    std::vector<float> got(out, 0.0f);
    ggml_gemm_q1_0_g128_4x4_q8_0(ctx.n_weights,
                                 got.data(), /* bs = */ (size_t) bs,
                                 ctx.packed_w.data(), q8_pack.data(),
                                 /* nr = */ nr, /* nc = */ ctx.n_w_rows);

    const float ref_mag = std::max(max_abs_value(ref), 1e-6f);
    const float tol     = 1e-3f * ref_mag + 1e-4f;
    int failed = 0;
    int max_print = 4;  // first 4 row-major elements per result
    for (size_t i = 0; i < out; i++) {
        const float diff = std::fabs(got[i] - ref[i]);
        const bool  ok   = diff <= tol;
        if (!ok || max_print > 0) {
            printf("  GEMM[nr=%d] elem %zu (row %zu, col %zu): ref=% .6f got=% .6f diff=% .6e %s\n",
                   nr, i, i / bs, i % bs,
                   (double) ref[i], (double) got[i], (double) diff,
                   ok ? "OK" : "FAIL");
            if (ok) max_print--;
        }
        failed += !ok;
    }
    if (failed) {
        printf("  GEMM[nr=%d]: %d/%zu elements out of tolerance\n",
               nr, failed, out);
    }
    return failed == 0;
}

} // namespace

int main(int /* argc */, char ** /* argv */) {
    ggml_cpu_init();

    std::mt19937 rng(0xC0FFEEu);

    test_context ctx;
    if (!prepare_weights(ctx, rng)) return 1;

    int failed = 0;

    printf("Test 1: GEMV (nr=1)\n");
    if (!run_gemv_test(ctx, rng)) failed++;

    printf("Test 2: GEMM (nr=4)\n");
    if (!run_gemm_test(ctx, /* nr = */ 4, rng)) failed++;

    printf("Test 3: GEMM (nr=8)\n");
    if (!run_gemm_test(ctx, /* nr = */ 8, rng)) failed++;

    if (failed) {
        printf("FAIL: %d/3 sub-tests failed\n", failed);
        return 1;
    }
    printf("PASS: GEMV (nr=1) + GEMM (nr=4) + GEMM (nr=8) all match single-row vec_dot within FP16 tolerance\n");
    return 0;
}
