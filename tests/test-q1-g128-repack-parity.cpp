// Parity test for the Phase 3 Q1_0_g128 4-row repack GEMV kernel.
//
// Background:
//   Phase 2 (PR #11) added an offline microbench harness for the proposed
//   AVX2 / AVX-VNNI repack kernels. Phase 3 (PR #12) ports the AVX2 path
//   into `ggml/src/ggml-cpu/arch/x86/repack.cpp` and wires it into the
//   runtime repack dispatcher in `repack.cpp`.
//
//   This test verifies that the production GEMV kernel (the one that
//   actually runs at inference time on AVX2-only hosts) produces FP32
//   output bit-identical (within FP16 quantisation noise) to the
//   single-row `vec_dot` already in mainline.
//
// Test design:
//   * 4 weight rows of `n` weights each, drawn from a deterministic
//     pseudo-random generator. n = 4 * QK1_0_g128 = 512 (the smallest
//     useful size).
//   * 1 activation column of `n` weights.
//   * Reference output: 4 calls to the public `ggml_get_type_traits_cpu(
//     GGML_TYPE_Q1_0_g128)->vec_dot` (single-row, mainline).
//   * Test output: 1 call to `ggml_gemv_q1_0_g128_4x4_q8_0` (4x4 GEMV
//     against a manually-repacked weight buffer).
//   * Pass criterion: per-row absolute error < 1e-3 of the reference
//     magnitude (FP16 round-trip + FMA reordering noise).
//
// The struct layout below MUST match `block_q1_0_g128x4` in
// `ggml/src/ggml-cpu/repack.h`. It is duplicated here only to avoid
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

// Forward-declare the production GEMV. Defined in
// `ggml/src/ggml-cpu/arch/x86/repack.cpp` (or `repack.cpp` for non-x86).
extern "C" void ggml_gemv_q1_0_g128_4x4_q8_0(int n, float * s, size_t bs,
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

float max_abs(const std::vector<float> & v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

} // namespace

int main(int /* argc */, char ** /* argv */) {
    ggml_cpu_init();

    constexpr int n_blocks_per_row = 4;     // 4 Q1_0_g128 blocks per row
    constexpr int n_weights = K_QK1_0_G128 * n_blocks_per_row; // 512
    constexpr int n_rows    = 4;            // GEMV processes 4 rows in parallel
    static_assert(n_weights % K_QK1_0_G128 == 0);
    static_assert(n_weights % K_QK8_0 == 0);

    const auto * traits_q1     = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0_g128);
    const auto * traits_q8_0   = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!traits_q1 || !traits_q1->from_float || !traits_q1->vec_dot ||
        !traits_q8_0 || !traits_q8_0->from_float) {
        printf("FAIL: cpu type traits missing for Q1_0_g128 / Q8_0\n");
        return 1;
    }
    if (traits_q1->vec_dot_type != GGML_TYPE_Q8_0) {
        printf("FAIL: expected Q1_0_g128 vec_dot_type == Q8_0, got %d\n",
               (int) traits_q1->vec_dot_type);
        return 1;
    }

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Synthetic weights: 4 rows × n_weights floats.
    std::vector<std::vector<float>> w(n_rows, std::vector<float>(n_weights));
    for (int r = 0; r < n_rows; r++) {
        for (int i = 0; i < n_weights; i++) w[r][i] = dist(rng);
    }
    // Synthetic activations: 1 column × n_weights floats.
    std::vector<float> y(n_weights);
    for (int i = 0; i < n_weights; i++) y[i] = dist(rng);

    // Quantise the 4 weight rows as Q1_0_g128 (single-row layout).
    std::vector<block_q1_0_g128_test> q1_rows(n_rows * n_blocks_per_row);
    for (int r = 0; r < n_rows; r++) {
        traits_q1->from_float(w[r].data(),
                              q1_rows.data() + r * n_blocks_per_row,
                              n_weights);
    }

    // Quantise the activation column as Q8_0.
    const int n_q8_blocks = n_weights / K_QK8_0;
    std::vector<block_q8_0_test> q8(n_q8_blocks);
    traits_q8_0->from_float(y.data(), q8.data(), n_weights);

    // Reference output: 4 calls to the single-row vec_dot.
    std::vector<float> ref(n_rows, 0.0f);
    for (int r = 0; r < n_rows; r++) {
        traits_q1->vec_dot(n_weights,
                           &ref[r], /* bs = */ 0,
                           q1_rows.data() + r * n_blocks_per_row, /* bx = */ 0,
                           q8.data(), /* by = */ 0,
                           /* nrc = */ 1);
    }

    // Repack 4 rows -> x4 layout.
    std::vector<block_q1_0_g128x4_test> packed(n_blocks_per_row);
    {
        block_q1_0_g128_test tmp4[4];
        for (int ib = 0; ib < n_blocks_per_row; ib++) {
            for (int r = 0; r < n_rows; r++) {
                tmp4[r] = q1_rows[r * n_blocks_per_row + ib];
            }
            repack_4_rows_to_x4(tmp4, &packed[ib]);
        }
    }

    // Test output: one GEMV call against the repacked buffer.
    std::vector<float> got(n_rows, 0.0f);
    ggml_gemv_q1_0_g128_4x4_q8_0(n_weights,
                                 got.data(), /* bs = */ 0,
                                 packed.data(), q8.data(),
                                 /* nr = */ 1, /* nc = */ n_rows);

    // Compare. FP16 round-trip + FMA reordering implies a worst-case
    // relative error around 2^-9; we use 1e-3 of the per-row magnitude
    // as a conservative absolute threshold.
    const float ref_mag = std::max(max_abs(ref), 1e-6f);
    const float tol     = 1e-3f * ref_mag + 1e-4f;
    int failed = 0;
    for (int r = 0; r < n_rows; r++) {
        const float diff = std::fabs(got[r] - ref[r]);
        const bool  ok   = diff <= tol;
        printf("row %d: ref=% .6f got=% .6f diff=% .6e tol=% .6e %s\n",
               r, (double) ref[r], (double) got[r], (double) diff,
               (double) tol, ok ? "OK" : "FAIL");
        failed += !ok;
    }

    if (failed) {
        printf("FAIL: %d/%d rows out of tolerance\n", failed, n_rows);
        return 1;
    }
    printf("PASS: all %d rows match single-row vec_dot within FP16 tolerance\n",
           n_rows);
    return 0;
}
