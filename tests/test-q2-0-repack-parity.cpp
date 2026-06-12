// Parity test for the Q2_0 4-row repack GEMV/GEMM kernels.
//
// Background:
//   Q2_0 (2-bit ternary, 128-weight blocks) shipped with a single-row
//   `vec_dot` only. This adds a 4-row interleaved runtime repack
//   (`block_q2_0x4`) plus AVX2 / AVX-VNNI GEMV and GEMM kernels in
//   `ggml/src/ggml-cpu/arch/x86/repack.cpp`, wired into the runtime repack
//   dispatcher in `repack.cpp`. It mirrors the Q1_0_g128 repack design;
//   see `tests/test-q1-g128-repack-parity.cpp` for the sibling test.
//
//   This test verifies that the production GEMV and GEMM kernels produce
//   FP32 output bit-identical (within FP16 quantisation noise) to the
//   single-row `vec_dot` already in mainline. GEMM is invoked by
//   `forward_mul_mat_one_chunk` whenever `nrows > 3`, with src1 pre-packed
//   as `block_q8_0x4` (4 activation rows interleaved); the test covers both
//   nr=1 (GEMV) and nr=4/8 (GEMM).
//
// Test design:
//   * 4 weight rows of `n_weights = 4 * QK2_0 = 512` weights each, drawn
//     from a deterministic pseudo-random generator. Quantised once as Q2_0
//     (single-row) and once via the manual repack helper into the
//     `block_q2_0x4` layout.
//   * For GEMV: 1 activation row of n_weights floats, quantised as plain
//     block_q8_0.
//   * For GEMM: nr activation rows, quantised as block_q8_0 (single-row)
//     for the reference, then packed into `block_q8_0x4` groups (mirror of
//     `ggml_quantize_mat_q8_0_4x4`) for the kernel input.
//   * Reference output: single-row `vec_dot` from
//     `ggml_get_type_traits_cpu(GGML_TYPE_Q2_0)`.
//   * Test output: `ggml_gemv_q2_0_4x4_q8_0` (nr=1) or
//     `ggml_gemm_q2_0_4x4_q8_0` (nr>=4).
//   * Pass criterion: per-element absolute error <= 1e-3 of the per-row
//     magnitude (FP16 round-trip + FMA reordering noise).
//
// The struct layouts below MUST match `block_q2_0x4` and `block_q8_0x4`
// in `ggml/src/ggml-cpu/repack.h`. They are duplicated here only to avoid
// adding the internal header to the test include path.

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

constexpr int K_QK2_0 = 128;
constexpr int K_QK8_0 = 32;

// Mirror of `block_q2_0x4` from `ggml/src/ggml-cpu/repack.h`.
struct block_q2_0x4_test {
    uint16_t d[4];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q2_0x4_test) == 8 + 128,
              "test layout must match ggml/src/ggml-cpu/repack.h");

// Mirror of `block_q2_0` from `ggml/src/ggml-common.h` — single-row layout
// the dispatcher repacks from.
struct block_q2_0_test {
    uint16_t d;
    uint8_t  qs[K_QK2_0 / 4];
};
static_assert(sizeof(block_q2_0_test) == 2 + K_QK2_0 / 4,
              "test single-row layout must match ggml-common.h");

// Mirror of `block_q8_0` from `ggml/src/ggml-common.h`.
struct block_q8_0_test {
    uint16_t d;
    int8_t   qs[K_QK8_0];
};
static_assert(sizeof(block_q8_0_test) == 34,
              "test q8_0 layout must match ggml-common.h");

// Mirror of `block<8, 4>` (= `block_q8_0x4`) from `repack.h`.
struct block_q8_0x4_test {
    uint16_t d[4];
    int8_t   qs[K_QK8_0 * 4];
};
static_assert(sizeof(block_q8_0x4_test) == 8 + K_QK8_0 * 4,
              "test q8_0x4 layout must match repack.h");

// Forward-declare the production GEMV / GEMM. Defined in
// `ggml/src/ggml-cpu/arch/x86/repack.cpp` (or `repack.cpp` for non-x86).
extern "C" void ggml_gemv_q2_0_4x4_q8_0(int n, float * s, size_t bs,
                                        const void * vx, const void * vy,
                                        int nr, int nc);
extern "C" void ggml_gemm_q2_0_4x4_q8_0(int n, float * s, size_t bs,
                                        const void * vx, const void * vy,
                                        int nr, int nc);

// Repack 4 single-row blocks into one x4 block, mirroring
// `make_block_q2_0x4` in `repack.cpp`. Q2_0 has 32 packed bytes per block
// (4 sub-blocks of 8 bytes); the x4 layout interleaves 8 bytes per row.
void repack_4_rows_to_x4(const block_q2_0_test in[4],
                         block_q2_0x4_test * out) {
    for (int r = 0; r < 4; r++) {
        out->d[r] = in[r].d;
    }
    for (int k = 0; k < 4; k++) {
        for (int r = 0; r < 4; r++) {
            std::memcpy(&out->qs[k * 32 + r * 8], &in[r].qs[k * 8], 8);
        }
    }
}

// Pack 4 quantised activation rows into a stream of `nb` `block_q8_0x4`,
// mirroring the encoder in `ggml_quantize_mat_q8_0_4x4_generic`.
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
    const ggml_type_traits_cpu * traits_q2   = nullptr;
    const ggml_type_traits_cpu * traits_q8_0 = nullptr;

    std::vector<std::vector<float>>     w;          // [n_w_rows][n_weights]
    std::vector<block_q2_0_test>        q2_rows;     // [n_w_rows * n_blocks]
    std::vector<block_q2_0x4_test>      packed_w;    // [n_blocks]
    int n_w_rows  = 0;
    int n_weights = 0;
    int n_blocks  = 0; // QK2_0 blocks per row
    int n_q8      = 0; // QK8_0 blocks per row
};

bool prepare_weights(test_context & ctx, std::mt19937 & rng) {
    ctx.traits_q2   = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
    ctx.traits_q8_0 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!ctx.traits_q2 || !ctx.traits_q2->from_float || !ctx.traits_q2->vec_dot ||
        !ctx.traits_q8_0 || !ctx.traits_q8_0->from_float) {
        printf("FAIL: cpu type traits missing for Q2_0 / Q8_0\n");
        return false;
    }
    if (ctx.traits_q2->vec_dot_type != GGML_TYPE_Q8_0) {
        printf("FAIL: expected Q2_0 vec_dot_type == Q8_0, got %d\n",
               (int) ctx.traits_q2->vec_dot_type);
        return false;
    }

    ctx.n_w_rows  = 4;
    ctx.n_blocks  = 4;
    ctx.n_weights = K_QK2_0 * ctx.n_blocks;
    ctx.n_q8      = ctx.n_weights / K_QK8_0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    ctx.w.assign(ctx.n_w_rows, std::vector<float>(ctx.n_weights));
    for (int r = 0; r < ctx.n_w_rows; r++) {
        for (int i = 0; i < ctx.n_weights; i++) ctx.w[r][i] = dist(rng);
    }

    ctx.q2_rows.assign((size_t) ctx.n_w_rows * ctx.n_blocks, {});
    for (int r = 0; r < ctx.n_w_rows; r++) {
        ctx.traits_q2->from_float(ctx.w[r].data(),
                                  ctx.q2_rows.data() + r * ctx.n_blocks,
                                  ctx.n_weights);
    }

    ctx.packed_w.assign(ctx.n_blocks, {});
    {
        block_q2_0_test tmp4[4];
        for (int ib = 0; ib < ctx.n_blocks; ib++) {
            for (int r = 0; r < ctx.n_w_rows; r++) {
                tmp4[r] = ctx.q2_rows[r * ctx.n_blocks + ib];
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
        ctx.traits_q2->vec_dot(ctx.n_weights,
                               &ref[r], /* bs = */ 0,
                               ctx.q2_rows.data() + r * ctx.n_blocks, /* bx = */ 0,
                               q8.data(), /* by = */ 0,
                               /* nrc = */ 1);
    }

    std::vector<float> got(ctx.n_w_rows, 0.0f);
    ggml_gemv_q2_0_4x4_q8_0(ctx.n_weights,
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

// Run a GEMM test case with `nr` activation rows. Output is laid out as a
// `nr × n_w_rows` matrix with the `bs` (float-element row stride) the
// production dispatcher passes; we choose `bs = n_w_rows`.
bool run_gemm_test(const test_context & ctx, int nr, std::mt19937 & rng) {
    assert(nr % 4 == 0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> y(nr, std::vector<float>(ctx.n_weights));
    for (int row = 0; row < nr; row++) {
        for (int i = 0; i < ctx.n_weights; i++) y[row][i] = dist(rng);
    }

    std::vector<std::vector<block_q8_0_test>> q8_per_row(
        nr, std::vector<block_q8_0_test>(ctx.n_q8));
    for (int row = 0; row < nr; row++) {
        ctx.traits_q8_0->from_float(y[row].data(),
                                    q8_per_row[row].data(),
                                    ctx.n_weights);
    }

    const int n_groups = nr / 4;
    std::vector<block_q8_0x4_test> q8_pack((size_t) n_groups * ctx.n_q8);
    for (int g = 0; g < n_groups; g++) {
        std::vector<block_q8_0_test> rows_arr[4] = {
            q8_per_row[g * 4 + 0], q8_per_row[g * 4 + 1],
            q8_per_row[g * 4 + 2], q8_per_row[g * 4 + 3],
        };
        pack_4_rows_to_q8_0x4(rows_arr, ctx.n_q8,
                              q8_pack.data() + (size_t) g * ctx.n_q8);
    }

    const int    bs  = ctx.n_w_rows;
    const size_t out = (size_t) nr * bs;
    std::vector<float> ref(out, 0.0f);
    for (int row = 0; row < nr; row++) {
        for (int r = 0; r < ctx.n_w_rows; r++) {
            ctx.traits_q2->vec_dot(ctx.n_weights,
                                   &ref[(size_t) row * bs + r], /* bs = */ 0,
                                   ctx.q2_rows.data() + r * ctx.n_blocks, /* bx = */ 0,
                                   q8_per_row[row].data(), /* by = */ 0,
                                   /* nrc = */ 1);
        }
    }

    std::vector<float> got(out, 0.0f);
    ggml_gemm_q2_0_4x4_q8_0(ctx.n_weights,
                            got.data(), /* bs = */ (size_t) bs,
                            ctx.packed_w.data(), q8_pack.data(),
                            /* nr = */ nr, /* nc = */ ctx.n_w_rows);

    const float ref_mag = std::max(max_abs_value(ref), 1e-6f);
    const float tol     = 1e-3f * ref_mag + 1e-4f;
    int failed = 0;
    int max_print = 4;
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
